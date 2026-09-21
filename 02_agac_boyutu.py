"""
02_agac_boyutu.py  —  Bedava Doğrulama Bütçesi
================================================================
micro-scale projesi

SORU
----
Spekülatif çözümlemede büyük model, taslağın önerdiği N pozisyonu tek
seferde kontrol eder. SpecExec bunun BEDAVA olduğunu varsayıyor: ağırlıklar
kaç pozisyon kontrol edersen et bir kere okunuyor.

Bu varsayım bir yere kadar doğru. Bir yerden sonra hesap yükü devreye
giriyor ve süre N ile birlikte artmaya başlıyor.

O nokta nerede? Buna "bedava bütçe" diyoruz.

Kartın zayıfsa bütçe küçük, güçlüyse büyük. SpecExec 2048 kullanıyor —
ama onlar 4090 ve A100'de ölçtü. 6 GB'lık bir kartta 2048 anlamlı mı?

BU ÖLÇÜM DONANIMA AİTTİR — Colab'da değil, kendi kartında çalıştır.

ÇIKTI
-----
sonuclar/csv/agac_<kart>_<zaman>.csv
sonuclar/grafik/agac_<kart>_<zaman>.png
"""

import os, gc, time, datetime
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================================
# AYARLAR
# =====================================================================

MODELLER = [
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    # Karta sığmazsa betik kendisi atlar, korkma.
    # ("Qwen/Qwen2.5-3B", "Qwen2.5-3B"),
]

N_LISTESI = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

ISINMA   = 3       # ölçüme sayılmayan deneme
TEKRAR   = 7       # medyan alınacak tekrar sayısı
DTYPE    = torch.float16

KLASOR_CSV    = "sonuclar/csv"
KLASOR_GRAFIK = "sonuclar/grafik"
ZAMAN = datetime.datetime.now().strftime("%Y%m%d_%H%M")


# =====================================================================
# KART BİLGİSİ
# =====================================================================

def kart_bilgisi():
    if not torch.cuda.is_available():
        return dict(ad="CPU", vram_gb=0.0, bant=0.0, pcie_gen=0, pcie_x=0)

    ozellik = torch.cuda.get_device_properties(0)
    ad = ozellik.name
    vram = ozellik.total_memory / 1e9

    try:
        gen = torch.cuda.get_device_properties(0).__dict__.get("pci_bus_id", None)
    except Exception:
        gen = None

    print(f"kart      : {ad}")
    print(f"VRAM      : {vram:.1f} GB")
    print(f"SM sayısı : {ozellik.multi_processor_count}")
    print(f"torch     : {torch.__version__}  CUDA {torch.version.cuda}")
    print()
    print("PCIe bilgisi için terminalde şunu çalıştır ve gunluk.txt'ye yapıştır:")
    print("  nvidia-smi --query-gpu=name,memory.total,"
          "pcie.link.width.current,pcie.link.gen.current --format=csv")
    print()

    return dict(ad=ad, vram_gb=vram, sm=ozellik.multi_processor_count)


# =====================================================================
# ÖLÇÜM
# =====================================================================

def sure_olc(model, n, cihaz, sozluk_boyu):
    """N pozisyonluk tek bir ileri geçişin medyan süresi (ms)."""
    girdi = torch.randint(0, sozluk_boyu, (1, n), device=cihaz)

    with torch.no_grad():
        for _ in range(ISINMA):
            model(girdi)
        if cihaz == "cuda":
            torch.cuda.synchronize()

        sureler = []
        for _ in range(TEKRAR):
            if cihaz == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(girdi)
            if cihaz == "cuda":
                torch.cuda.synchronize()
            sureler.append((time.perf_counter() - t0) * 1000)

    return float(np.median(sureler)), float(np.std(sureler))


def modeli_olc(model_id, model_ad, kart, satirlar):
    print(f"\n{'='*62}\n{model_ad}\n{'='*62}")
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"

    tok   = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE if cihaz == "cuda" else torch.float32
    ).to(cihaz).eval()

    n_param = sum(p.numel() for p in model.parameters())
    sozluk  = model.config.vocab_size
    print(f"  {n_param/1e9:.2f} milyar parametre, ~{n_param*2/1e9:.2f} GB (fp16)")

    if cihaz == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for n in N_LISTESI:
        try:
            ms, sapma = sure_olc(model, n, cihaz, sozluk)
        except torch.cuda.OutOfMemoryError:
            print(f"  N={n:<5} bellek yetmedi, duruldu")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  N={n:<5} hata: {type(e).__name__}: {e}")
            break

        tepe = torch.cuda.max_memory_allocated()/1e9 if cihaz == "cuda" else 0.0

        satirlar.append(dict(
            kart       = kart["ad"],
            model      = model_ad,
            n_param    = n_param,
            N          = n,
            ms         = ms,
            sapma_ms   = sapma,
            ms_basina_pozisyon = ms / n,
            tepe_gb    = tepe,
        ))
        print(f"  N={n:<5} {ms:8.2f} ms   (±{sapma:.2f})   "
              f"pozisyon başına {ms/n:7.3f} ms")

    del model
    gc.collect()
    if cihaz == "cuda":
        torch.cuda.empty_cache()


# =====================================================================
# BEDAVA BÜTÇE
# =====================================================================

def bedava_butce(alt, esik=1.25):
    """
    N=1'e göre süresi <= esik katı olan en büyük N.
    Yani "hâlâ neredeyse bedava" sayılan doğrulama genişliği.
    """
    taban = alt[alt.N == 1].ms.iloc[0]
    uygun = alt[alt.ms <= taban * esik]
    return int(uygun.N.max()) if len(uygun) else 1


# =====================================================================
# GRAFİK
# =====================================================================

def grafik(df, kart_ad, yol):
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
    renkler = plt.cm.tab10(np.linspace(0, .5, df.model.nunique()))

    for renk, m in zip(renkler, df.model.unique()):
        alt = df[df.model == m].sort_values("N")
        taban = alt[alt.N == 1].ms.iloc[0]
        b = bedava_butce(alt)

        # 1) Toplam süre
        ax[0].plot(alt.N, alt.ms, "o-", lw=2, ms=5, color=renk, label=m)
        ax[0].axhline(taban, color=renk, ls=":", lw=1, alpha=.6)

        # 2) Pozisyon başına süre — asıl anlatan grafik
        ax[1].plot(alt.N, alt.ms_basina_pozisyon, "o-", lw=2, ms=5,
                   color=renk, label=f"{m}  (bedava bütçe ≈ {b})")
        ax[1].axvline(b, color=renk, ls="--", lw=1.2, alpha=.7)

        # 3) Doğrulama kapasitesi
        ax[2].plot(alt.N, alt.N / (alt.ms / 1000), "o-", lw=2, ms=5,
                   color=renk, label=m)

    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    ax[0].set_xlabel("tek turda doğrulanan pozisyon  (N)")
    ax[0].set_ylabel("bir turun süresi (ms)")
    ax[0].set_title("1 — Tur süresi\nyatay nokta çizgi = N=1 seviyesi")
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)

    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
    ax[1].set_xlabel("tek turda doğrulanan pozisyon  (N)")
    ax[1].set_ylabel("pozisyon başına ms")
    ax[1].set_title("2 — Pozisyon başına maliyet\n"
                    "düz inen kısım = bedava bölge, dipten sonrası = hesaba takıldın")
    ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)

    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("N"); ax[2].set_ylabel("saniyede doğrulanan pozisyon")
    ax[2].set_title("3 — Doğrulama kapasitesi\ntepe noktası = en verimli N")
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)

    fig.suptitle(f"Bedava doğrulama bütçesi — {kart_ad}", y=1.02, fontsize=12)
    plt.tight_layout(); plt.savefig(yol, dpi=150, bbox_inches="tight"); plt.close()


# =====================================================================

def main():
    os.makedirs(KLASOR_CSV, exist_ok=True)
    os.makedirs(KLASOR_GRAFIK, exist_ok=True)

    kart = kart_bilgisi()
    kart_kisa = kart["ad"].replace(" ", "_").replace("NVIDIA_", "")

    satirlar = []
    for mid, mad in MODELLER:
        try:
            modeli_olc(mid, mad, kart, satirlar)
        except Exception as e:
            print(f"  ! {mad} atlandı: {type(e).__name__}: {e}")

        if satirlar:
            pd.DataFrame(satirlar).to_csv(
                f"{KLASOR_CSV}/agac_{kart_kisa}_{ZAMAN}.csv", index=False)

    if not satirlar:
        print("\nHiç ölçüm yapılamadı.")
        return

    df = pd.DataFrame(satirlar)
    grafik(df, kart["ad"], f"{KLASOR_GRAFIK}/agac_{kart_kisa}_{ZAMAN}.png")

    print("\n" + "="*62)
    print("ÖZET")
    print("="*62)
    print(f"{'model':<16}{'bedava bütçe':>14}{'en verimli N':>14}{'kapasite':>12}")
    print("-"*62)
    for m in df.model.unique():
        alt = df[df.model == m].sort_values("N")
        b = bedava_butce(alt)
        kap = alt.N / (alt.ms/1000)
        en_iyi = int(alt.N.iloc[int(np.argmax(kap.values))])
        print(f"{m:<16}{b:>14}{en_iyi:>14}{kap.max():>12.0f} poz/s")

    print("\nYORUM:")
    print("  bedava bütçe, tek turda ~bedava doğrulayabildiğin pozisyon sayısı.")
    print("  SpecExec 128–2048 kullanıyor (RTX 4090 / A100).")
    print("  Senin bütçen bunun çok altındaysa, büyük ağaç stratejisi")
    print("  bu kartta işlemez — ve bu tek başına bir bulgudur.")
    print(f"\nCSV    : {KLASOR_CSV}/agac_{kart_kisa}_{ZAMAN}.csv")
    print(f"GRAFİK : {KLASOR_GRAFIK}/agac_{kart_kisa}_{ZAMAN}.png")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntoplam süre: {time.time()-t0:.0f} sn")
