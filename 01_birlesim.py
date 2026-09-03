"""
03_birlesim.py  —  Birleşim Seyrelmesi Deneyi
================================================================
micro-scale projesi

SORU
----
Aktivasyon seyrekliği KELİME BAŞINA çalışır: her kelime için nöronların
küçük bir kısmı "uyanık" sayılır.

Spekülatif çözümleme ise BİRDEN ÇOK KELİMEYİ AYNI ANDA doğrular. Yani k
kelimenin ihtiyaç duyduğu nöronların BİRLEŞİMİ getirilmek zorundadır.

Eğer ardışık kelimeler hep aynı nöronları uyandırıyorsa birleşim küçük
kalır ve iki yöntem güzelce birleşir. Eğer her kelime farklı nöronları
uyandırıyorsa birleşim şişer ve seyreklik erir.

Bu betik o eğriyi ölçer.

KARŞILAŞTIRMA NOKTASI (analitik)
--------------------------------
Nöronlar her kelimede rastgele seçilseydi, p oranında seyreklikte
k kelimenin birleşimi tam olarak şu olurdu:

    beklenen_birlesim = 1 - (1 - p)^k

p=0.20, k=4  ->  0.590

Ölçtüğümüz değer bu sayıya yakınsa kümeler bağımsız demektir (kötü).
p'ye yakınsa kümeler neredeyse aynı demektir (mükemmel).

ÇIKTI
-----
sonuclar/csv/birlesim_<model>_<zaman>.csv
sonuclar/grafik/birlesim_<model>_<zaman>.png

NOT: Sonuç dosyalarının üzerine asla yazılmaz, isimde zaman damgası vardır.
"""

import os
import gc
import json
import time
import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

MODELLER = [
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("HuggingFaceTB/SmolLM2-360M", "SmolLM2-360M"),
    ("gpt2", "GPT-2"),
]

# Seyreklik seviyeleri: nöronların yüzde kaçını "uyanık" sayıyoruz
SEYREKLIK = [0.10, 0.20, 0.30, 0.50]

# Kaç ardışık kelimenin birleşimine bakılacak
K_LISTESI = [1, 2, 3, 4, 6, 8]

SEQ_LEN     = 512      # bir seferde işlenen kelime sayısı
N_PENCERE   = 4        # kaç metin parçası işlenecek
CIHAZ       = "cuda" if torch.cuda.is_available() else "cpu"

# Türkçe ölçüm için buraya bir .txt yolu ver. None ise WikiText-2 kullanılır.
METIN_DOSYASI = None
# METIN_DOSYASI = "veri/turkce.txt"

KLASOR_CSV     = "sonuclar/csv"
KLASOR_GRAFIK  = "sonuclar/grafik"

ZAMAN = datetime.datetime.now().strftime("%Y%m%d_%H%M")


# =====================================================================
# METİN HAZIRLAMA
# =====================================================================

YEDEK_METIN = (
    "The transformer architecture processes sequences of tokens through "
    "stacked layers of attention and feed forward networks. Each layer reads "
    "from and writes to a shared residual stream. "
) * 400


def metni_getir():
    """Değerlendirme metnini döndürür."""
    if METIN_DOSYASI and os.path.exists(METIN_DOSYASI):
        with open(METIN_DOSYASI, encoding="utf-8") as f:
            metin = f.read()
        print(f"  metin: {METIN_DOSYASI} ({len(metin)} karakter)")
        return metin

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        metin = "\n\n".join(ds["text"])
        print(f"  metin: WikiText-2 test ({len(metin)} karakter)")
        return metin
    except Exception as e:
        print(f"  ! WikiText yüklenemedi ({e}), yedek metin kullanılıyor")
        return YEDEK_METIN


# =====================================================================
# MLP KATMANLARINI BULMA
# =====================================================================

def mlp_cikis_katmanlari(model):
    """
    MLP'nin son projeksiyonunu (down_proj / mlp.c_proj) döndürür.
    Bu katmanın GİRDİSİ, nöron aktivasyonlarının ta kendisidir.

    Modern modeller (Qwen, Llama, SmolLM): ...mlp.down_proj
    GPT-2                                : ...mlp.c_proj

    DİKKAT: GPT-2'de attn.c_proj de var, onu almamak lazım.
    """
    bulunan = []
    for ad, modul in model.named_modules():
        if ".mlp." not in ad and not ad.endswith("mlp"):
            continue
        if ad.endswith("down_proj") or ad.endswith("mlp.c_proj"):
            bulunan.append((ad, modul))

    if not bulunan:
        raise RuntimeError("MLP çıkış katmanı bulunamadı — mimari tanınmadı")
    return bulunan


# =====================================================================
# AKTİVASYON YAKALAMA
# =====================================================================

class Yakalayici:
    """
    down_proj'un girdisini yakalar ve her kelime pozisyonu için
    en büyük |değer|'e sahip nöronların indekslerini saklar.

    Aktivasyonların kendisini saklamıyoruz — sadece sıralamayı.
    Böylece hafıza yükü düşük kalıyor.
    """

    def __init__(self, max_oran):
        self.max_oran = max_oran
        self.indeksler = {}   # katman adı -> (seq, tut) int16 tensörü
        self.n_noron   = {}

    def kanca(self, ad):
        def _f(modul, girdi):
            akt = girdi[0]                      # (batch, seq, n_noron)
            if akt.dim() == 3:
                akt = akt[0]                    # batch=1 varsayımı
            n_noron = akt.shape[-1]
            tut = max(1, int(n_noron * self.max_oran))

            # En büyük mutlak değerli nöronların indeksleri, büyükten küçüğe
            _, idx = torch.topk(akt.abs().float(), k=tut, dim=-1)

            self.indeksler[ad] = idx.to(torch.int16).cpu()
            self.n_noron[ad]   = n_noron
        return _f


# =====================================================================
# BİRLEŞİM HESABI
# =====================================================================

def birlesim_orani(idx, n_noron, tut, k):
    """
    idx     : (seq, max_tut) int16, her pozisyon için sıralı nöron indeksleri
    tut     : bu seyreklik seviyesinde kaç nöron uyanık sayılıyor
    k       : kaç ardışık kelimenin birleşimine bakılıyor

    Dönen: bütün pencereler üzerinden ortalama birleşim oranı
    """
    seq = idx.shape[0]
    if seq < k:
        return float("nan")

    kesik = idx[:, :tut].long()                       # (seq, tut)

    # Her pozisyon için bool maske
    maske = torch.zeros(seq, n_noron, dtype=torch.bool)
    maske.scatter_(1, kesik, True)

    # Kayan pencerelerde birleşim
    oranlar = []
    for t in range(seq - k + 1):
        birlesim = maske[t:t + k].any(dim=0).sum().item()
        oranlar.append(birlesim / n_noron)

    return float(np.mean(oranlar))


def ardisik_ortusme(idx, tut):
    """
    Ardışık iki kelimenin kümelerinin Jaccard benzerliği.
    1.0 = tamamen aynı, 0.0 = hiç ortak nöron yok.
    """
    seq = idx.shape[0]
    kesik = idx[:, :tut].long()
    skorlar = []
    for t in range(seq - 1):
        a = set(kesik[t].tolist())
        b = set(kesik[t + 1].tolist())
        skorlar.append(len(a & b) / len(a | b))
    return float(np.mean(skorlar))


# =====================================================================
# ANA DÖNGÜ
# =====================================================================

def modeli_isle(model_id, model_ad, metin, satirlar):
    print(f"\n{'=' * 60}\n{model_ad}\n{'=' * 60}")

    tok   = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if CIHAZ == "cuda" else torch.float32,
    ).to(CIHAZ).eval()

    katmanlar = mlp_cikis_katmanlari(model)
    print(f"  {len(katmanlar)} MLP katmanı bulundu")

    yakalayici = Yakalayici(max_oran=max(SEYREKLIK))
    tutamaclar = [m.register_forward_pre_hook(yakalayici.kanca(a))
                  for a, m in katmanlar]

    tokenlar = tok(metin, return_tensors="pt").input_ids[0]
    print(f"  {len(tokenlar)} token")

    for pencere in range(N_PENCERE):
        bas = pencere * SEQ_LEN
        son = bas + SEQ_LEN
        if son > len(tokenlar):
            break

        parca = tokenlar[bas:son].unsqueeze(0).to(CIHAZ)

        yakalayici.indeksler.clear()
        with torch.no_grad():
            model(parca)

        for kat_i, (kat_ad, _) in enumerate(katmanlar):
            idx     = yakalayici.indeksler[kat_ad]
            n_noron = yakalayici.n_noron[kat_ad]

            for p in SEYREKLIK:
                tut = max(1, int(n_noron * p))

                ortusme = ardisik_ortusme(idx, tut)

                for k in K_LISTESI:
                    olculen  = birlesim_orani(idx, n_noron, tut, k)
                    rastgele = 1.0 - (1.0 - p) ** k

                    satirlar.append(dict(
                        model         = model_ad,
                        katman        = kat_i,
                        katman_ad     = kat_ad,
                        n_noron       = n_noron,
                        pencere       = pencere,
                        seyreklik     = p,
                        k             = k,
                        birlesim      = olculen,
                        rastgele      = rastgele,
                        # 0 = kümeler aynı, 1 = kümeler bağımsız
                        bagimsizlik   = (olculen - p) / max(rastgele - p, 1e-9),
                        ardisik_jaccard = ortusme,
                    ))

        print(f"  pencere {pencere + 1}/{N_PENCERE} bitti")

    for t in tutamaclar:
        t.remove()
    del model
    gc.collect()
    if CIHAZ == "cuda":
        torch.cuda.empty_cache()


# =====================================================================
# GRAFİK
# =====================================================================

def grafik_ciz(df, model_ad, yol):
    fig, eksenler = plt.subplots(1, 2, figsize=(13, 5))

    # --- Sol: birleşim eğrisi ---
    ax = eksenler[0]
    for p in SEYREKLIK:
        alt = df[(df.model == model_ad) & (df.seyreklik == p)]
        if alt.empty:
            continue
        ort = alt.groupby("k")["birlesim"].mean()
        ax.plot(ort.index, ort.values, "o-", lw=2, label=f"ölçülen p={p}")

        rast = [1 - (1 - p) ** k for k in ort.index]
        ax.plot(ort.index, rast, "--", lw=1, alpha=0.5, color="gray")

    ax.set_xlabel("aynı anda doğrulanan kelime sayısı (k)")
    ax.set_ylabel("birleşim kümesinin oranı")
    ax.set_title(f"{model_ad} — birleşim seyrelmesi\n"
                 "kesikli gri = rastgele olsaydı")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # --- Sağ: katman katman bağımsızlık ---
    ax = eksenler[1]
    for p in SEYREKLIK:
        alt = df[(df.model == model_ad) & (df.seyreklik == p) & (df.k == 4)]
        if alt.empty:
            continue
        ort = alt.groupby("katman")["bagimsizlik"].mean()
        ax.plot(ort.index, ort.values, "o-", lw=1.5, ms=3, label=f"p={p}")

    ax.axhline(0, color="green", ls=":", lw=1)
    ax.axhline(1, color="red", ls=":", lw=1)
    ax.set_xlabel("katman")
    ax.set_ylabel("bağımsızlık  (0 = aynı nöronlar, 1 = rastgele)")
    ax.set_title(f"{model_ad} — katman katman (k=4)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(yol, dpi=140)
    plt.close()


# =====================================================================
# ÇALIŞTIR
# =====================================================================

def main():
    os.makedirs(KLASOR_CSV, exist_ok=True)
    os.makedirs(KLASOR_GRAFIK, exist_ok=True)

    print(f"cihaz: {CIHAZ}")
    if CIHAZ == "cuda":
        print(f"kart : {torch.cuda.get_device_name(0)}")

    metin = metni_getir()
    satirlar = []

    for model_id, model_ad in MODELLER:
        try:
            modeli_isle(model_id, model_ad, metin, satirlar)
        except Exception as e:
            print(f"  ! {model_ad} atlandı: {type(e).__name__}: {e}")
            continue

        # Her modelden sonra kaydet — çökerse veri kaybolmasın
        df = pd.DataFrame(satirlar)
        csv_yol = f"{KLASOR_CSV}/birlesim_{ZAMAN}.csv"
        df.to_csv(csv_yol, index=False)

        try:
            grafik_ciz(df, model_ad,
                       f"{KLASOR_GRAFIK}/birlesim_{model_ad}_{ZAMAN}.png")
        except Exception as e:
            print(f"  ! grafik çizilemedi: {e}")

    if not satirlar:
        print("\nHiçbir model işlenemedi.")
        return

    df = pd.DataFrame(satirlar)
    print(f"\nCSV: {KLASOR_CSV}/birlesim_{ZAMAN}.csv")

    # --- Özet ---
    print("\n" + "=" * 70)
    print("ÖZET  —  p = 0.20, k = 4")
    print("=" * 70)
    print(f"{'model':<16} {'ölçülen':>9} {'rastgele':>9} "
          f"{'bağımsızlık':>12} {'jaccard':>9}")
    print("-" * 70)

    for model_ad in df.model.unique():
        alt = df[(df.model == model_ad) & (df.seyreklik == 0.20) & (df.k == 4)]
        if alt.empty:
            continue
        print(f"{model_ad:<16} "
              f"{alt.birlesim.mean():>9.3f} "
              f"{alt.rastgele.mean():>9.3f} "
              f"{alt.bagimsizlik.mean():>12.3f} "
              f"{alt.ardisik_jaccard.mean():>9.3f}")

    print("\nYORUM:")
    print("  bağımsızlık 0'a yakın -> ardışık kelimeler aynı nöronları")
    print("     kullanıyor. Seyreklik + spekülatif çözümleme BİRLEŞİR.")
    print("  bağımsızlık 1'e yakın -> her kelime farklı nöron istiyor.")
    print("     Seyreklik, spekülatif çözümleme altında ERİR.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntoplam süre: {time.time() - t0:.0f} sn")
