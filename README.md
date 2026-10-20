# Klasifikasi Penggerak Utama Deforestasi

Proyek ini merupakan studi kasus sederhana penerapan *machine learning* untuk
mengklasifikasikan penyebab utama (*direct drivers*) dari gangguan tutupan hutan.
Fitur sintetisnya mensimulasikan pengamatan Sentinel-1 (radar), Sentinel-2
(optik), serta konteks spasial.

Studi kasus ini terinspirasi materi webinar *Informatics Practitioner Lecture
Series* bersama Dr. Bart Slaughter (Wageningen University & Research). Dataset
yang dihasilkan oleh proyek ini sepenuhnya sintetis dan bukan pengukuran satelit
atau data lapangan sebenarnya.

## Tujuan

- Mengklasifikasikan tujuh kategori penggerak gangguan hutan.
- Membandingkan Random Forest standar dengan Random Forest yang memakai
  `class_weight="balanced"`.
- Mengamati dampak ketidakseimbangan kelas melalui metrik keseluruhan dan
  per kelas.

## Dataset sintetis

Skrip membuat 1.000 sampel secara default. Fitur yang digunakan:

| Fitur | Deskripsi |
| --- | --- |
| `s1_vv_backscatter` | Simulasi nilai backscatter VV Sentinel-1 |
| `s1_vh_backscatter` | Simulasi nilai backscatter VH Sentinel-1 |
| `s2_ndvi` | Simulasi indeks vegetasi NDVI Sentinel-2 |
| `s2_ndwi` | Simulasi indeks air NDWI Sentinel-2 |
| `patch_size_ha` | Luas area gangguan (hektar) |
| `dist_to_road_km` | Jarak ke jalan terdekat (kilometer) |
| `driver_label` | Label numerik kelas (0–6) |
| `driver_name` | Nama kelas |

Distribusi target yang dimaksudkan: Smallholder Agriculture (35%), Large-scale
Agriculture (25%), Road Development (10%), Selective Logging (10%), Mining
(5%), Wildfire (10%), dan Flooding (5%). Jumlah aktual per kelas dapat berbeda
karena sampel diambil secara acak.

| Label | Kelas |
| ---: | --- |
| 0 | Smallholder Ag |
| 1 | Large Ag |
| 2 | Road Dev |
| 3 | Selective Logging |
| 4 | Mining |
| 5 | Wildfire |
| 6 | Flooding |

## Metode dan evaluasi

Data dibagi menjadi set pelatihan (80%) dan pengujian (20%) dengan stratifikasi
kelas. Fitur tidak diskalakan karena Random Forest tidak memerlukan feature
scaling. Kedua model dievaluasi menggunakan:

- **Accuracy**
- **Micro F1-Score** — untuk klasifikasi multikelas single-label, nilainya
  sama dengan accuracy.
- **Macro F1-Score** — rata-rata F1 tiap kelas dengan bobot yang sama.
- **Classification report** dan **confusion matrix** untuk model balanced.

`class_weight="balanced"` mengubah bobot kelas saat pelatihan, tetapi tidak
menjamin peningkatan performa pada setiap kelas. Hasil eksperimen sintetis ini
tidak dapat dianggap sebagai validasi untuk pemantauan deforestasi nyata.

## Teknologi

- Python 3.x
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

## Menjalankan proyek

1. Buka folder proyek di VS Code.
2. Pasang dependensi:

   ```powershell
   python -m pip install numpy pandas scikit-learn matplotlib seaborn
   ```

3. Jalankan skrip:

   ```powershell
   python deforestation_driver_classification.py
   ```

Skrip mencetak distribusi kelas dan metrik evaluasi ke terminal, lalu membuat
`deforestation_data.csv` dan `confusion_matrix.png` pada folder kerja saat
skrip dijalankan.

## Struktur proyek

```text
deforestation-driver-analysis/
├── deforestation_driver_classification.py
├── README.md
├── deforestation_data.csv  # dibuat saat skrip dijalankan
└── confusion_matrix.png    # dibuat saat skrip dijalankan
```

## Referensi

Slaughter, B. (2026). *Attributing Direct Drivers to Satellite-based
Deforestation Alerts using Artificial Intelligence*. Informatics Practitioner
Lecture Series, Wageningen University & Research.
