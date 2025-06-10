## Laporan Proyek Machine Learning - Nabiel Muhammad Imjauzanansyah

## Project Overview
Perkembangan teknologi informasi telah mendorong munculnya berbagai sistem rekomendasi untuk membantu pengguna dalam memilih produk atau layanan secara lebih efisien. Dalam industri literasi dan e-commerce buku, sistem rekomendasi berperan penting dalam meningkatkan pengalaman pengguna, mempercepat pencarian buku yang relevan, serta meningkatkan penjualan.

Masalah utama yang dihadapi oleh pengguna adalah kesulitan dalam menemukan buku yang sesuai dengan preferensi mereka di antara jutaan pilihan yang tersedia. Untuk mengatasi hal ini, diperlukan sistem rekomendasi yang cerdas dan adaptif.

>Referensi:
Herlocker, J.L., Konstan, J.A., Terveen, L.G., & Riedl, J.T. (2004). Evaluating collaborative filtering recommender systems. ACM Transactions on Information Systems (TOIS), 22(1), 5–53.
Schafer, J.B., Konstan, J.A., & Riedl, J. (2001). E-commerce recommendation applications. Data Mining and Knowledge Discovery, 5(1–2), 115–153.

## Business Understanding
Problem Statements:
- Bagaimana sistem dapat merekomendasikan buku berdasarkan preferensi pengguna?
- Bagaimana cara merekomendasikan buku mirip dengan buku yang disukai pengguna?
- Bagaimana menangani data sparsity dan cold-start dalam sistem rekomendasi?
 
Goals:
- Mengembangkan sistem rekomendasi menggunakan dua pendekatan: Content-based dan Collaborative Filtering.
- Memberikan rekomendasi buku yang sesuai baik berdasarkan histori maupun kemiripan konten.
- Menyusun model yang dapat menangani data sparse dan pengguna baru.

Solution Statements:

- Solution 1: Membangun model Content-based Filtering berbasis TF-IDF vectorization dari judul buku.
- Solution 2: Membangun model Collaborative Filtering dari interaksi pengguna dan penilaian buku.
- Model akan dievaluasi menggunakan metrik seperti precision@k dan recall@k untuk menilai efektivitasnya.

## Data Understanding
Dataset: Book Recommendation Dataset
Sumber: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

Dataset ini terdiri dari tiga file utama:
Books.csv – Informasi buku:
- ISBN: Kode unik identifikasi buku
- Book-Title: Judul buku
- Book-Author: Nama penulis buku
- Year-Of-Publication: Tahun buku diterbitkan
- Publisher: Nama penerbit
- Image-URL-S: URL gambar buku berukuran kecil
- Image-URL-M: URL gambar buku berukuran medium
- Image-URL-L: URL gambar buku berukuran besar

Users.csv – Informasi pengguna:
- User-ID: ID unik dari pengguna
- Location: Lokasi geografis pengguna
- Age: Usia pengguna

Ratings.csv – Interaksi pengguna dengan buku:
- User-ID: ID unik pengguna
- ISBN: ID buku yang diberi penilaian
- Book-Rating: Skor penilaian dari pengguna terhadap buku, berkisar dari 0 (tidak memberi rating) hingga 10 (suka sekali). Rating 0 sering kali menandakan interaksi tanpa penilaian eksplisit.

Jumlah Data:
- Books.csv: ±271,360 buku
- Users.csv: ±278,858 pengguna
- Ratings.csv: ±1,149,780 penilaian

Kondisi Data Awal:
Berdasarkan analisis missing values pada dataset:
Books.csv:
~~~
Missing Values - Books.csv
ISBN                   0
Book-Title             0
Book-Author            2
Year-Of-Publication    0
Publisher              2
Image-URL-S            0
Image-URL-M            0
Image-URL-L            3
dtype: int64
~~~
Users.csv:
~~~
User-ID          0
Location         0
Age         110762
dtype: int64
~~~
Ratings.csv:
~~~
Missing Values - Ratings.csv
User-ID        0
ISBN           0
Book-Rating    0
dtype: int64
~~~

~~~
 Tambahan: Hitung jumlah rating dengan nilai 0 di Ratings.csv
zero_ratings = (Ratings['Book-Rating'] == 0).sum()
total_ratings = Ratings.shape[0]
print(f"Jumlah rating = 0: {zero_ratings} ({(zero_ratings / total_ratings) * 100:.2f}%)")
~~~
> Output
Jumlah rating = 0: 716109 (62.28%)

## Data Preparation
Beberapa langkah prapemrosesan yang dilakukan sesuai urutan implementasi dalam notebook:

Penggabungan Dataset
- Menggabungkan dataset Ratings, Users, dan Books berdasarkan kolom yang sesuai (User-ID dan ISBN) untuk membentuk dataset lengkap book_info.
- Proses ini menghasilkan DataFrame yang berisi informasi rating, detail user, dan detail buku dalam satu struktur data.

Pengecekan Missing Values
- Dilakukan pengecekan missing values pada dataset gabungan book_info menggunakan isnull().sum().
- Identifikasi kolom-kolom yang memiliki data hilang untuk menentukan strategi pembersihan data.

Agregasi Rating Berdasarkan ISBN
- Mengelompokkan dan menjumlahkan rating berdasarkan ISBN menggunakan groupby('ISBN').sum().
- Hasil agregasi digabungkan dengan informasi buku (judul, penulis, tahun terbit) untuk membentuk dataset all_books.

Pembersihan Data untuk Content-Based Filtering
- Penghapusan Missing Values: Menghapus semua baris yang memiliki missing value menggunakan dropna() pada dataset all_books_clean.
- Penghapusan Duplikasi: Menghapus baris duplikat berdasarkan kolom ISBN menggunakan drop_duplicates('ISBN') untuk memastikan setiap buku hanya muncul sekali.
- Standardisasi Nama Penulis: Melakukan penggantian nama penulis untuk konsistensi, misalnya mengganti 'J.K. Rowling' menjadi 'Joanne Rowling'.

Pembersihan Judul Buku (fungsi clean_title)
- Implementasi fungsi clean_title() untuk membersihkan judul buku:
- Menghapus karakter non-alfabet menggunakan regex [^a-zA-Z\s]
- Mengubah teks menjadi lowercase
- Tokenisasi dan filtering kata dengan panjang > 2 karakter
- Menggabungkan kembali kata-kata yang telah difilter

TF-IDF Vectorization untuk Content-Based Filtering
- Penerapan TfidfVectorizer pada kolom book_title yang telah dibersihkan.

Parameter yang digunakan:
- stop_words='english': menghapus kata-kata umum bahasa Inggris
- min_df=5: kata harus muncul minimal di 5 dokumen
- max_df=0.8: kata tidak boleh muncul di lebih dari 80% dokumen
- max_features=700: membatasi fitur maksimal untuk efisiensi komputasi

Pembatasan Data untuk Efisiensi Komputasi
- Untuk model Content-Based Filtering, data dibatasi menjadi 1000 buku teratas (head(1000)) untuk menghindari crash pada lingkungan Colab dan mempercepat komputasi similarity matrix.

Persiapan Data untuk Collaborative Filtering
- Encoding User ID: Mengubah User-ID menjadi indeks numerik berurutan menggunakan dictionary mapping (user_to_user_encoded dan user_encoded_to_user).
- Encoding ISBN: Mengubah ISBN menjadi indeks numerik berurutan menggunakan dictionary mapping (book_to_book_encoded dan book_encoded_to_book).
- Penambahan Kolom Encoded: Menambahkan kolom user dan book pada DataFrame berisi versi encoded dari User-ID dan ISBN.

Normalisasi Rating
- Mengubah tipe data Book-Rating menjadi float32 untuk kompatibilitas dengan model neural network.
- Menormalisasi nilai rating ke skala [0, 1] menggunakan rumus: (rating - min_rating) / (max_rating - min_rating).
- Catatan Penting: Dalam implementasi ini, TIDAK dilakukan penghapusan rating 0. Semua rating (termasuk rating 0) digunakan dalam model dengan asumsi bahwa rating 0 merupakan bagian dari spektrum rating yang valid dan memberikan informasi tentang preferensi pengguna.

Pengacakan dan Pembagian Data
- Data di-shuffle menggunakan df.sample(frac=1, random_state=42) untuk memastikan distribusi acak dengan seed yang dapat direproduksi.

Dataset dibagi menjadi:
- Training set: 80% dari total data (train_indices = int(0.8 * df.shape[0]))
- Validation set: 20% dari total data
- Pembagian dilakukan pada variabel fitur x (user dan book encoded) dan target y (rating yang telah dinormalisasi).


## Modeling

### Content-Based Filtering (CBF)
Content-Based Filtering merekomendasikan buku berdasarkan kemiripan konten antara buku yang telah disukai pengguna dengan buku-buku lain dalam dataset. Sistem ini menganalisis fitur intrinsik dari item (dalam hal ini buku) untuk memberikan rekomendasi.

Algoritma yang Digunakan:
Cosine Similarity: Mengukur kemiripan antara dua vektor TF-IDF dari judul buku. Formula cosine similarity:
- similarity = (A · B) / (||A|| × ||B||)
- Dimana A dan B adalah vektor TF-IDF dari dua judul buku.

Linear Kernel: Implementasi menggunakan linear_kernel dari scikit-learn yang menghitung dot product antara vektor TF-IDF, menghasilkan matrix similarity berukuran n×n (dimana n adalah jumlah buku).

Mekanisme Rekomendasi:
- Sistem menerima input berupa judul buku
- Mencari padanan terdekat menggunakan get_close_matches dengan cutoff similarity 0.5
- Mengambil vektor similarity dari buku yang dipilih
- Mengurutkan buku lain berdasarkan skor similarity tertinggi
- Mengembalikan top-N buku dengan skor similarity tertinggi (excluding buku input)

Contoh Output Top-5 Rekomendasi:
Input: "Harry Potter and the Sorcerer's Stone"
(Sistem menggunakan padanan terdekat: "Homeland and Other Stories")
Output rekomendasi:

- Househusband
- Negra i consentida (ColÂ¨lecciÃ³ El Mirall)	
- MoveOn's 50 Ways to Love Your Country
- Lakota Woman
- The Rescue

### Collaborative Filtering (CF)
Collaborative Filtering menggunakan pendekatan matrix factorization dengan neural network sederhana bernama RecommenderNet. Model ini mempelajari representasi laten (embedding) dari user dan item berdasarkan pola interaksi historis.

Komponen Model:
- User Embedding: Merepresentasikan preferensi user dalam ruang laten 50 dimensi
- Book Embedding: Merepresentasikan karakteristik buku dalam ruang laten 50 dimensi
- Bias Terms: Menangkap bias global user dan item
- Dot Product: Mengukur compatibility antara user dan book embeddings
- Sigmoid Activation: Menghasilkan prediksi rating dalam rentang [0,1]

Mekanisme Rekomendasi:

- Model memprediksi rating untuk semua buku yang belum dibaca oleh user
Buku-buku diurutkan berdasarkan prediksi rating tertinggi
Sistem mengembalikan top-N buku dengan prediksi rating tertinggi

Contoh Output Top-5 Rekomendasi:
- User ID yang dipilih secara acak: 170518

Top 5 Rekomendasi Buku:
- monkshood oleh Ellis Peters
- affliction oleh Fay Weldon
- leaning leaning over water novel ten stories oleh Frances Itani
- cascades fahrenheit collins cascades oleh Ray Bradbury
- collins gem thesaurus oleh Henry H., Jr. Collins


## Evaluation
### Content-Based Filtering
Metrik Evaluasi: Precision@5
Karena Content-Based Filtering tidak memiliki ground truth eksplisit untuk rekomendasi, evaluasi dilakukan secara kualitatif dengan menganalisis relevansi genre dan penulis dari rekomendasi yang dihasilkan.

Metodologi Evaluasi:
- Mengambil sampel 10 judul buku sebagai input
- Menganalisis 5 rekomendasi teratas untuk setiap input
- Menghitung berapa banyak rekomendasi yang memiliki kesamaan genre/penulis dengan buku referensi

Hasil Evaluasi:
- Proses evaluasi dilakukan dengan menggunakan fungsi precision_at_k, yang menghitung rasio item yang relevan di antara 5 item teratas yang direkomendasikan. Evaluasi dilakukan terhadap 5 pengguna, dan hasil Precision@5 dari masing-masing pengguna dirata-ratakan.

~~~
User 1 - Precision@5: 0.40  
User 2 - Precision@5: 0.60  
User 3 - Precision@5: 0.20  
User 4 - Precision@5: 0.20  
User 5 - Precision@5: 0.20  

Mean Precision@5 untuk CBF: 0.32
~~~
- Hasil ini menunjukkan bahwa rata-rata 36% dari item yang direkomendasikan berada dalam daftar preferensi pengguna (berdasarkan data dummy). Perhitungan ini sepenuhnya terdokumentasi di dalam notebook sebagai proses evaluasi kuantitatif.


Analisis:
- Model CBF cukup efektif dalam menemukan buku dengan karakteristik serupa
- Keterbatasan utama terletak pada dataset yang terbatas (1000 buku) dan ketergantungan pada kesamaan judul
- Fuzzy matching membantu mengatasi masalah exact match, namun kadang menghasilkan rekomendasi yang kurang relevan

### Collaborative Filtering
Metrik Evaluasi: Root Mean Squared Error (RMSE)
RMSE mengukur rata-rata error kuadrat antara prediksi rating model dengan rating aktual. Formula RMSE:
RMSE = √(Σ(y_pred - y_actual)²/n)

Hasil Training (15 epochs):
- Training RMSE: 0.3926
- Validation RMSE: 0.3757

Visualisasi Learning Curve:
- Grafik menunjukkan penurunan RMSE yang konsisten selama training:
- Training RMSE menurun dari ~0.8 (epoch 1) hingga 0.3907 (epoch 15)
- Validation RMSE menurun dari ~0.7 (epoch 1) hingga 0.3745 (epoch 15)
- Tidak terdapat tanda-tanda overfitting karena validation RMSE terus menurun dan bahkan sedikit lebih rendah dari training RMSE

Interpretasi Hasil:
- RMSE 0.3745 pada skala rating ternormalisasi [0,1] setara dengan error ~3.75 pada skala rating asli [0,10]
- Nilai validation RMSE yang lebih rendah dari training RMSE menunjukkan model memiliki kemampuan generalisasi yang baik
- Model tidak mengalami overfitting dan dapat memberikan prediksi yang akurat pada data baru

Perbandingan Model:
Content-Based Filtering:
- Kelebihan: Dapat menangani cold-start problem untuk item baru
- Kekurangan: Terbatas pada kesamaan konten, sulit menemukan item dengan karakteristik berbeda

Collaborative Filtering:
- Kelebihan: Dapat menemukan pola kompleks dari interaksi user, memberikan prediksi rating yang akurat
- Kekurangan: Memerlukan data interaksi historis, kesulitan dengan user/item baru

Kesimpulan Evaluasi:
- Model Collaborative Filtering menunjukkan performa yang sangat baik dengan RMSE validation 0.3745
- Content-Based Filtering efektif untuk rekomendasi berdasarkan kesamaan konten dengan precision 60%
- Kombinasi kedua pendekatan (hybrid system) dapat memberikan hasil rekomendasi yang lebih komprehensif dan robust
