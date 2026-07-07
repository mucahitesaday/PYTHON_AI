"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         VEKTÖR VERİTABANLARI — Kapsamlı Rehber ve Kod Örnekleri            ║
║                 ChromaDB (Çalışır) · Milvus (Üretim Referansı)             ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. VEKTÖR VERİTABANI NEDİR?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Geleneksel veritabanları tam eşleşme arar (WHERE isim = 'Ali').
Vektör veritabanları ANLAM benzerliği arar.

Nasıl çalışır?
  ┌─────────────────┐    Embedding    ┌──────────────────────┐
  │  "Kedi besleme" │ ──────Model───► │ [0.2, -0.8, 0.5, ...]│ (1536 boyut)
  └─────────────────┘                 └──────────────────────┘
                                               │
                                    Vektör Uzayına Yerleştir
                                               │
                                    ┌──────────▼──────────┐
                                    │  "Hayvan bakımı"    │ ◄── Yakın!
                                    │  "Köpek eğitimi"    │ ◄── Yakın!
                                    │  "Borsa analizi"    │    Uzak
                                    └─────────────────────┘

2. KULLANIM ALANLARI
━━━━━━━━━━━━━━━━━━━━
  • RAG (Retrieval-Augmented Generation) — LLM'e bağlam sağlama
  • Semantik Arama               — Anlam tabanlı arama motoru
  • Öneri Sistemleri             — "Buna benzeyenleri göster"
  • Görüntü/Ses Benzerliği       — İçerik tabanlı medya arama
  • Anomali Tespiti              — Anormal vektörleri bulma
  • Duplicate Tespiti            — Benzer belgeleri bulma

3. VEKTÖRLERİ KARŞILAŞTIRMA YÖNTEMLERİ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Kosinüs Benzerliği  → Yön farkı ölçer (NLP için ideal)
  Öklid Mesafesi (L2) → Mutlak uzaklık ölçer (görüntü için ideal)
  İç Çarpım (IP)      → Büyüklük + yön (öneri sistemleri için)

  cosine_sim(A, B) = (A·B) / (|A| × |B|)
  Sonuç: -1 (zıt) ile 1 (aynı) arasında

4. YAKLAŞIK KOMŞULuk ARAMA (ANN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Milyarlarca vektörde tam arama (brute-force) çok yavaştır.
  ANN algoritmaları hızlı ama yaklaşık sonuç verir:

  HNSW  (Hierarchical Navigable Small World) → Grafik tabanlı
  IVF   (Inverted File Index)                → Kümeleme tabanlı
  ANNOY (Spotify)                            → Ağaç tabanlı
  ScaNN (Google)                             → Kuantizasyon tabanlı

  ChromaDB varsayılan: HNSW
  Milvus varsayılan: IVF_FLAT → IVF_SQ8 → HNSW (ölçeğe göre)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARAÇLAR KARŞILAŞTIRMASI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ChromaDB:
    + Python native, kolay kurulum (pip install chromadb)
    + Yerleşik embedding (sentence-transformers entegrasyonu)
    + Metadata filtreleme desteği
    + In-memory veya dosya tabanlı
    - Milyar ölçeğinde performans sınırlı
    Kullanım: Prototip, orta ölçek, RAG uygulamaları

  Milvus:
    + Milyar ölçeğinde vektör (petabyte)
    + Dağıtık mimari (Kubernetes destekli)
    + Gelişmiş indeks türleri (20+)
    + Atamik işlem desteği
    - Kurulum karmaşık (Docker/K8s gerektirir)
    Kullanım: Büyük ölçek üretim, e-ticaret arama, sosyal medya

  Pinecone:   Bulut yönetimli, sıfır operasyon yükü
  Weaviate:   GraphQL + semantik arama + otomatik şema
  Qdrant:     Rust tabanlı, Rust/Python/Go client
  pgvector:   PostgreSQL eklentisi, SQL + vektör birlikte
"""

import chromadb
import numpy as np
import math
import time
import json
import hashlib
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BÖLÜM 1: VEKTÖR MATEMATİĞİ — Temel Kavramlar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VektorMatematigi:
    """
    Vektör veritabanlarının altında yatan matematik.
    Gerçek uygulamalarda bu işlemleri kütüphaneler yapar,
    ama kavramsal anlamak için kendi implementasyonumuz.
    """

    @staticmethod
    def kosinüs_benzerligi(a: List[float], b: List[float]) -> float:
        """
        İki vektör arasındaki kosinüs benzerliğini hesaplar.

        Formül: cos(θ) = (A·B) / (|A| × |B|)
        Sonuç aralığı: -1.0 (tamamen zıt) ile 1.0 (tamamen aynı)

        NLP'de neden kosinüs?
        Çünkü "ev" ve "evler" aynı yönü işaret eder, sadece büyüklükleri
        farklı olabilir. Kosinüs yönü ölçer, büyüklüğü değil.
        """
        a_arr = np.array(a)
        b_arr = np.array(b)

        dot_product = np.dot(a_arr, b_arr)          # Nokta çarpımı
        norm_a = np.linalg.norm(a_arr)               # |A| büyüklüğü
        norm_b = np.linalg.norm(b_arr)               # |B| büyüklüğü

        if norm_a == 0 or norm_b == 0:
            return 0.0  # Sıfır vektör kontrolü

        return float(dot_product / (norm_a * norm_b))

    @staticmethod
    def oklid_mesafesi(a: List[float], b: List[float]) -> float:
        """
        İki vektör arasındaki Öklid (L2) mesafesini hesaplar.

        Formül: d(A,B) = √(Σ(aᵢ - bᵢ)²)
        Sonuç: 0 (aynı nokta) ile ∞ arasında

        Görüntü benzerliğinde neden L2?
        Pikseller arasındaki gerçek uzaklık önemlidir.
        Renk/doku değerleri büyüklük bilgisi taşır.
        """
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.linalg.norm(a_arr - b_arr))

    @staticmethod
    def ic_carpim(a: List[float], b: List[float]) -> float:
        """
        İç çarpım (dot product) benzerliği.

        Formül: A·B = Σ(aᵢ × bᵢ)
        Hem yönü hem büyüklüğü dikkate alır.

        Öneri sistemlerinde neden IP?
        Kullanıcı puanlamaları büyüklük bilgisi taşır.
        Yüksek puan verilen öğelerin büyüklüğü yüksek olur.
        """
        return float(np.dot(np.array(a), np.array(b)))

    @staticmethod
    def normalize_et(vektor: List[float]) -> List[float]:
        """
        Vektörü birim vektöre dönüştürür (büyüklük = 1).
        Normalize edilmiş vektörlerde iç çarpım = kosinüs benzerliği.
        """
        v = np.array(vektor)
        norm = np.linalg.norm(v)
        if norm == 0:
            return vektor
        return (v / norm).tolist()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BÖLÜM 2: EMBEDDING MOTORU — Metin → Vektör Dönüşümü
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmbeddingMotoru:
    """
    Metni sabit boyutlu vektöre dönüştürür.

    GERÇEK ÜRETIM EMBEDDİNGLERİ:
    ┌────────────────────────────┬────────┬─────────────────────────────┐
    │ Model                      │ Boyut  │ Kullanım                    │
    ├────────────────────────────┼────────┼─────────────────────────────┤
    │ text-embedding-3-small     │ 1536   │ OpenAI, genel amaç          │
    │ text-embedding-3-large     │ 3072   │ OpenAI, yüksek doğruluk     │
    │ all-MiniLM-L6-v2           │ 384    │ HuggingFace, hızlı          │
    │ all-mpnet-base-v2          │ 768    │ HuggingFace, dengeli        │
    │ paraphrase-multilingual    │ 768    │ Çok dilli, Türkçe destekli  │
    │ BAAI/bge-large-en-v1.5     │ 1024   │ En yüksek benchmark         │
    └────────────────────────────┴────────┴─────────────────────────────┘

    Bu sınıf: Kurulum gerektirmeyen deterministik simülasyon.
    Gerçek benzerlik ilişkilerini koruyacak şekilde tasarlandı.
    """

    # Türkçe anlam grupları — benzer kelimelerin yakın vektör üretmesi için
    ANLAM_GRUPLARI = {
        "teknoloji": ["python", "kod", "yazılım", "programlama", "algoritma",
                      "yapay zeka", "makine öğrenmesi", "derin öğrenme", "ai",
                      "api", "veritabanı", "framework", "geliştirici"],
        "sağlık":    ["hastalık", "tedavi", "doktor", "hastane", "ilaç",
                      "sağlık", "tıp", "hasta", "klinik", "tanı", "semptom"],
        "finans":    ["para", "yatırım", "borsa", "hisse", "kripto", "bitcoin",
                      "ekonomi", "faiz", "döviz", "banka", "finans", "fiyat"],
        "yemek":     ["yemek", "tarif", "pişir", "malzeme", "mutfak", "lezzet",
                      "restoran", "aşçı", "yemek", "kahvaltı", "akşam"],
        "spor":      ["futbol", "basketbol", "koş", "antrenman", "spor",
                      "maç", "takım", "oyuncu", "gol", "kazanmak"],
        "eğitim":    ["okul", "öğrenci", "ders", "öğretmen", "eğitim",
                      "üniversite", "sınav", "öğrenmek", "akademi", "kurs"],
    }

    def __init__(self, boyut: int = 64):
        """
        boyut: Üretilecek embedding vektörünün boyutu.
        Gerçekte 384-3072 arası; burada demo için 64.
        """
        self.boyut = boyut
        self._cache: Dict[str, List[float]] = {}  # Performans için önbellek

    def _kelime_grubu_skoru(self, metin: str) -> Dict[str, float]:
        """Metnin her anlam grubuna ait skorunu hesaplar."""
        metin_lower = metin.lower()
        skorlar = {}
        for grup, kelimeler in self.ANLAM_GRUPLARI.items():
            skor = sum(1.0 for k in kelimeler if k in metin_lower)
            # Tam kelime eşleşmesine bonus
            skor += sum(0.5 for k in kelimeler if k == metin_lower)
            skorlar[grup] = skor
        return skorlar

    def encode(self, metin: str) -> List[float]:
        """
        Metni embedding vektörüne dönüştürür.

        Algoritma:
        1. Anlam grubu tespiti (anlamsal bileşen)
        2. Hash tabanlı deterministik gürültü (benzersizlik)
        3. Normalizasyon (birim vektör)

        Gerçek modellerde: Transformer attention mekanizması
        """
        if metin in self._cache:
            return self._cache[metin]

        metin_lower = metin.lower()
        vektor = np.zeros(self.boyut)

        # ─── Bileşen 1: Anlam Grubu Tabanlı Kodlama ──────────────────────
        # Her anlam grubuna vektör uzayında farklı bir bölge tahsis et.
        # Benzer anlamlı metinler aynı bölgelerde kümelensin.
        gruplar = list(self.ANLAM_GRUPLARI.keys())
        grup_boyutu = self.boyut // len(gruplar)

        anlam_skorlari = self._kelime_grubu_skoru(metin_lower)
        for i, (grup, skor) in enumerate(anlam_skorlari.items()):
            if skor > 0:
                baslangic = i * grup_boyutu
                bitis = baslangic + grup_boyutu
                # Gruba ait boyutları anlamsal skorla doldur
                grup_vektoru = np.array([
                    skor * math.sin(j * 0.7 + i * 1.2)
                    for j in range(bitis - baslangic)
                ])
                vektor[baslangic:bitis] += grup_vektoru

        # ─── Bileşen 2: Hash Tabanlı Benzersizlik ────────────────────────
        # Aynı anlam grubundaki farklı metinlerin biraz farklı olması için
        hash_bytes = hashlib.sha256(metin.encode()).digest()
        hash_vektor = np.array([
            ((b / 127.5) - 1.0) * 0.25  # [-0.25, 0.25] aralığında gürültü
            for b in hash_bytes[:self.boyut]
        ])
        vektor[:len(hash_vektor)] += hash_vektor

        # ─── Bileşen 3: Karakter N-gram Özelliği ─────────────────────────
        # Kelime yapısını yakalayan özellik (prefix, suffix vb.)
        for i in range(0, len(metin_lower) - 1):
            bigram = metin_lower[i:i+2]
            bigram_hash = int(hashlib.md5(bigram.encode()).hexdigest()[:4], 16)
            idx = bigram_hash % self.boyut
            vektor[idx] += 0.1

        # ─── Normalizasyon: Kosinüs benzerliği için birim vektör ─────────
        norm = np.linalg.norm(vektor)
        if norm > 0:
            vektor = vektor / norm

        sonuc = vektor.tolist()
        self._cache[metin] = sonuc
        return sonuc

    def toplu_encode(self, metinler: List[str]) -> List[List[float]]:
        """Birden fazla metni vektöre dönüştürür (batch processing)."""
        return [self.encode(metin) for metin in metinler]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BÖLÜM 3: CHROMADB — Detaylı Örnekler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChromaDBOrnek:
    """
    ChromaDB ile kapsamlı vektör veritabanı örnekleri.

    ChromaDB Mimarisi:
    ┌────────────────────────────────────────────────────┐
    │                   ChromaDB Client                  │
    ├────────────────────────────────────────────────────┤
    │  Collection 1   │  Collection 2  │  Collection N   │
    │  (belgeler)     │  (görseller)   │  (kullanıcılar) │
    ├─────────────────┴────────────────┴─────────────────┤
    │         HNSW İndeksi (Yaklaşık KNN Arama)          │
    ├────────────────────────────────────────────────────┤
    │  SQLite (Metadata)  │  Parquet (Vektörler)         │
    └────────────────────────────────────────────────────┘

    Collection = Tablo gibi düşünün.
    Her kayıt şunları içerir:
      - id          : Benzersiz tanımlayıcı (string)
      - embedding   : Vektör (sayı listesi)
      - document    : Ham metin (opsiyonel)
      - metadata    : Ek bilgiler (dict) — filtrelemede kullanılır
    """

    def __init__(self):
        # ─── İstemci Başlatma ─────────────────────────────────────────────
        # EphemeralClient: Sadece bellekte, process bitince kaybolur
        # PersistentClient: Diske kaydeder, kalıcı
        # HttpClient:       Uzak ChromaDB sunucusuna bağlanır
        self.client = chromadb.EphemeralClient()

        self.embedding = EmbeddingMotoru(boyut=64)
        self.matematik = VektorMatematigi()
        print("✅ ChromaDB başlatıldı (EphemeralClient — bellekte)")

    # ─────────────────────────────────────────────────────────────────────────
    # ÖRNEK 1: Koleksiyon Yönetimi
    # ─────────────────────────────────────────────────────────────────────────

    def ornek_1_koleksiyon_yonetimi(self):
        """
        ChromaDB'de koleksiyon oluşturma, silme ve yönetme.

        Koleksiyon = RDBMS'deki tablo gibi
        distance_function seçimi performans ve doğruluğu etkiler.
        """
        print("\n" + "═" * 60)
        print("  ÖRNEK 1: Koleksiyon Yönetimi")
        print("═" * 60)

        # ─── Kosinüs Benzerliği İle Koleksiyon ────────────────────────────
        # NLP görevleri için önerilen — yön tabanlı karşılaştırma
        belge_koleksiyonu = self.client.create_collection(
            name="belgeler",
            metadata={
                "hnsw:space": "cosine",     # Mesafe fonksiyonu: cosine, l2, ip
                "hnsw:M": 16,               # Graf bağlantı sayısı (16-64 arası)
                "hnsw:construction_ef": 200, # İndeks oluşturma hassasiyeti
                "hnsw:search_ef": 100,      # Arama hassasiyeti (runtime)
                "açıklama": "Şirket belgeleri ve dökümanlar"
            }
        )
        print(f"  ✓ '{belge_koleksiyonu.name}' koleksiyonu oluşturuldu")
        print(f"    Mesafe fonksiyonu: cosine (NLP için ideal)")

        # ─── L2 Mesafesi İle Koleksiyon ────────────────────────────────────
        # Görüntü, ses gibi büyüklük bilgisi önemli olan veriler için
        goruntu_koleksiyonu = self.client.create_collection(
            name="gorseller",
            metadata={
                "hnsw:space": "l2",         # Öklid mesafesi
                "açıklama": "Görüntü feature vektörleri"
            }
        )
        print(f"  ✓ '{goruntu_koleksiyonu.name}' koleksiyonu oluşturuldu")
        print(f"    Mesafe fonksiyonu: l2 (görüntü için ideal)")

        # ─── Mevcut Koleksiyonları Listele ─────────────────────────────────
        koleksiyonlar = self.client.list_collections()
        print(f"\n  📋 Toplam koleksiyon sayısı: {len(koleksiyonlar)}")
        for kol in koleksiyonlar:
            print(f"     • {kol}")

        # ─── get_or_create: Varsa al, yoksa oluştur ────────────────────────
        # Üretim kodunda genellikle bu pattern kullanılır
        kullanici_kol = self.client.get_or_create_collection(
            name="kullanicilar",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"\n  ✓ get_or_create: '{kullanici_kol.name}' hazır")

        return belge_koleksiyonu

    # ─────────────────────────────────────────────────────────────────────────
    # ÖRNEK 2: Veri Ekleme (Add / Upsert)
    # ─────────────────────────────────────────────────────────────────────────

    def ornek_2_veri_ekleme(self, koleksiyon):
        """
        ChromaDB'ye vektör ve metadata ekleme yöntemleri.

        add():    Yeni kayıt ekler, ID zaten varsa HATA
        upsert(): Varsa günceller, yoksa ekler (üretimde tercih edilir)
        """
        print("\n" + "═" * 60)
        print("  ÖRNEK 2: Veri Ekleme")
        print("═" * 60)

        # ─── Örnek Veri Seti: Şirket Belgeleri ────────────────────────────
        belgeler = [
            {
                "id": "DOC001",
                "metin": "Python programlama dili makine öğrenmesi için idealdir",
                "meta": {"kategori": "teknoloji", "dil": "python", "yil": 2024,
                         "yazar": "Ali Veli", "güvenilirlik": 0.95}
            },
            {
                "id": "DOC002",
                "metin": "Yapay zeka derin öğrenme ile görüntü tanıma yapabilir",
                "meta": {"kategori": "teknoloji", "dil": "python", "yil": 2024,
                         "yazar": "Ayşe Kaya", "güvenilirlik": 0.90}
            },
            {
                "id": "DOC003",
                "metin": "Sinir ağları büyük veri setleri üzerinde eğitilir",
                "meta": {"kategori": "teknoloji", "dil": "python", "yil": 2023,
                         "yazar": "Mehmet Can", "güvenilirlik": 0.88}
            },
            {
                "id": "DOC004",
                "metin": "Diyabet hastalığının belirtileri ve tedavi yöntemleri",
                "meta": {"kategori": "sağlık", "dil": "turkce", "yil": 2024,
                         "yazar": "Dr. Fatma", "güvenilirlik": 0.99}
            },
            {
                "id": "DOC005",
                "metin": "Kalp hastalıkları için sağlıklı beslenme rehberi",
                "meta": {"kategori": "sağlık", "dil": "turkce", "yil": 2023,
                         "yazar": "Dr. Ahmet", "güvenilirlik": 0.97}
            },
            {
                "id": "DOC006",
                "metin": "Borsa yatırımı için teknik analiz stratejileri",
                "meta": {"kategori": "finans", "dil": "turkce", "yil": 2024,
                         "yazar": "Borsacı Ali", "güvenilirlik": 0.75}
            },
            {
                "id": "DOC007",
                "metin": "Kripto para Bitcoin ve Ethereum yatırım rehberi",
                "meta": {"kategori": "finans", "dil": "turkce", "yil": 2024,
                         "yazar": "Kripto Uzmanı", "güvenilirlik": 0.70}
            },
            {
                "id": "DOC008",
                "metin": "Türk mutfağı geleneksel yemek tarifleri koleksiyonu",
                "meta": {"kategori": "yemek", "dil": "turkce", "yil": 2023,
                         "yazar": "Şef Hüseyin", "güvenilirlik": 0.92}
            },
            {
                "id": "DOC009",
                "metin": "Vegan beslenme ve bitkisel protein kaynakları",
                "meta": {"kategori": "yemek", "dil": "turkce", "yil": 2024,
                         "yazar": "Diyetisyen", "güvenilirlik": 0.93}
            },
            {
                "id": "DOC010",
                "metin": "Futbol antrenman programı ve kondisyon egzersizleri",
                "meta": {"kategori": "spor", "dil": "turkce", "yil": 2024,
                         "yazar": "Antrenör Murat", "güvenilirlik": 0.85}
            },
        ]

        # ─── Embedding Üret ────────────────────────────────────────────────
        print("  📐 Embedding vektörleri oluşturuluyor...")
        embeddingler = []
        for belge in belgeler:
            vec = self.embedding.encode(belge["metin"])
            embeddingler.append(vec)

        # ─── Toplu Ekleme (Batch Add) ──────────────────────────────────────
        # add() — ID çakışması olursa Exception fırlatır
        # Üretimde upsert() tercih edin
        koleksiyon.add(
            ids       = [b["id"] for b in belgeler],
            embeddings = embeddingler,
            documents  = [b["metin"] for b in belgeler],
            metadatas  = [b["meta"] for b in belgeler],
        )
        print(f"  ✓ {len(belgeler)} belge eklendi")

        # ─── Tek Kayıt Upsert ──────────────────────────────────────────────
        # Varsa günceller, yoksa ekler — idempotent operasyon
        yeni_belge_id = "DOC011"
        yeni_metin = "Makine öğrenmesi algoritmaları karşılaştırması"
        koleksiyon.upsert(
            ids        = [yeni_belge_id],
            embeddings = [self.embedding.encode(yeni_metin)],
            documents  = [yeni_metin],
            metadatas  = [{"kategori": "teknoloji", "dil": "python",
                           "yil": 2024, "yazar": "Test", "güvenilirlik": 0.80}]
        )
        print(f"  ✓ Upsert: '{yeni_belge_id}' eklendi/güncellendi")

        # ─── Koleksiyon Durumu ─────────────────────────────────────────────
        toplam = koleksiyon.count()
        print(f"\n  📊 Koleksiyondaki toplam kayıt: {toplam}")

        return belgeler

    # ─────────────────────────────────────────────────────────────────────────
    # ÖRNEK 3: Semantik Arama
    # ─────────────────────────────────────────────────────────────────────────

    def ornek_3_semantik_arama(self, koleksiyon):
        """
        Anlam tabanlı benzerlik araması.

        query_embeddings: Arama vektörü (kendimiz embedding ürettik)
        query_texts:      ChromaDB embedding fonksiyonu ayarlıysa kullanılır
        n_results:        Kaç sonuç dönsün
        include:          Sonuçta hangi alanlar olsun
        """
        print("\n" + "═" * 60)
        print("  ÖRNEK 3: Semantik Arama")
        print("═" * 60)

        sorgular = [
            "derin öğrenme ve yapay zeka",
            "sağlıklı yaşam ve hastalık önleme",
            "hisse senedi ve para yatırımı",
        ]

        for sorgu in sorgular:
            print(f"\n  🔍 Sorgu: '{sorgu}'")
            print(f"  {'─' * 50}")

            # ─── Sorgu Embedding'i Üret ────────────────────────────────────
            sorgu_vektoru = self.embedding.encode(sorgu)

            # ─── ChromaDB Sorgusu ──────────────────────────────────────────
            # include listesi: documents, metadatas, distances, embeddings
            # distances: küçük değer = daha benzer (cosine space'de)
            sonuclar = koleksiyon.query(
                query_embeddings = [sorgu_vektoru],
                n_results        = 3,           # En benzer 3 sonuç
                include          = ["documents", "metadatas", "distances"]
            )

            # ─── Sonuçları İşle ve Göster ──────────────────────────────────
            for i, (doc, meta, dist) in enumerate(zip(
                sonuclar["documents"][0],
                sonuclar["metadatas"][0],
                sonuclar["distances"][0]
            )):
                # Cosine distance → similarity dönüşümü
                # ChromaDB: distance = 1 - cosine_similarity
                benzerlk = 1.0 - dist
                print(f"  {i+1}. [Benzerlik: {benzerlk:.3f}] {doc[:55]}...")
                print(f"      Kategori: {meta['kategori']} | "
                      f"Güvenilirlik: {meta['güvenilirlik']}")

    # ─────────────────────────────────────────────────────────────────────────
    # ÖRNEK 4: Metadata Filtrelemeli Arama (Hybrid Search)
    # ─────────────────────────────────────────────────────────────────────────

    def ornek_4_metadata_filtreleme(self, koleksiyon):
        """
        Vektör aramasını metadata koşulları ile birleştirme.

        Bu "Hybrid Search" olarak adlandırılır:
        1. WHERE koşulları önce uygulanır (metadata filter)
        2. Kalan kayıtlar içinde vektör araması yapılır

        ChromaDB where operatörleri:
          $eq    → eşit
          $ne    → eşit değil
          $gt    → büyüktür
          $gte   → büyüktür veya eşit
          $lt    → küçüktür
          $lte   → küçüktür veya eşit
          $in    → listede var
          $nin   → listede yok
          $and   → VE koşulu
          $or    → VEYA koşulu
        """
        print("\n" + "═" * 60)
        print("  ÖRNEK 4: Metadata Filtrelemeli Arama")
        print("═" * 60)

        sorgu_vektoru = self.embedding.encode("programlama ve algoritma")

        # ─── Filtre 1: Kategori Eşleşmesi ─────────────────────────────────
        print("\n  📌 Filtre 1: Sadece 'teknoloji' kategorisi")
        sonuc1 = koleksiyon.query(
            query_embeddings = [sorgu_vektoru],
            n_results        = 3,
            where            = {"kategori": {"$eq": "teknoloji"}},
            include          = ["documents", "metadatas", "distances"]
        )
        for doc, dist in zip(sonuc1["documents"][0], sonuc1["distances"][0]):
            print(f"  → [{1-dist:.3f}] {doc[:60]}...")

        # ─── Filtre 2: Güvenilirlik Eşiği ──────────────────────────────────
        print("\n  📌 Filtre 2: Güvenilirlik > 0.90")
        sonuc2 = koleksiyon.query(
            query_embeddings = [sorgu_vektoru],
            n_results        = 3,
            where            = {"güvenilirlik": {"$gt": 0.90}},
            include          = ["documents", "metadatas", "distances"]
        )
        for doc, meta, dist in zip(
            sonuc2["documents"][0],
            sonuc2["metadatas"][0],
            sonuc2["distances"][0]
        ):
            print(f"  → [{1-dist:.3f}] {doc[:50]}... "
                  f"(güvenilirlik: {meta['güvenilirlik']})")

        # ─── Filtre 3: AND Koşulu ───────────────────────────────────────────
        print("\n  📌 Filtre 3: teknoloji VE yil=2024")
        sonuc3 = koleksiyon.query(
            query_embeddings = [sorgu_vektoru],
            n_results        = 3,
            where            = {
                "$and": [
                    {"kategori": {"$eq": "teknoloji"}},
                    {"yil":      {"$eq": 2024}}
                ]
            },
            include = ["documents", "distances"]
        )
        print(f"  Bulunan sonuç: {len(sonuc3['documents'][0])}")
        for doc, dist in zip(sonuc3["documents"][0], sonuc3["distances"][0]):
            print(f"  → [{1-dist:.3f}] {doc[:60]}...")

        # ─── Filtre 4: OR Koşulu ────────────────────────────────────────────
        print("\n  📌 Filtre 4: kategori = 'sağlık' VEYA 'spor'")
        saglik_vektoru = self.embedding.encode("sağlıklı yaşam egzersiz")
        sonuc4 = koleksiyon.query(
            query_embeddings = [saglik_vektoru],
            n_results        = 4,
            where            = {
                "$or": [
                    {"kategori": {"$eq": "sağlık"}},
                    {"kategori": {"$eq": "spor"}}
                ]
            },
            include = ["documents", "metadatas", "distances"]
        )
        for doc, meta, dist in zip(
            sonuc4["documents"][0],
            sonuc4["metadatas"][0],
            sonuc4["distances"][0]
        ):
            print(f"  → [{1-dist:.3f}] [{meta['kategori']}] {doc[:50]}...")

        # ─── Filtre 5: IN Operatörü ─────────────────────────────────────────
        print("\n  📌 Filtre 5: $in operatörü ile çoklu ID")
        sonuc5 = koleksiyon.get(
            ids     = ["DOC001", "DOC004", "DOC008"],
            include = ["documents", "metadatas"]
        )
        print(f"  Belirli ID'lerle get(): {len(sonuc5['documents'])} kayıt")
        for doc in sonuc5["documents"]:
            print(f"  → {doc[:60]}...")

    # ─────────────────────────────────────────────────────────────────────────
    # ÖRNEK 5: Belge Güncelleme ve Silme
    # ─────────────────────────────────────────────────────────────────────────

    def ornek_5_guncelleme_silme(self, koleksiyon):
        """
        Koleksiyondaki kayıtları güncelleme ve silme işlemleri.
        """
        print("\n" + "═" * 60)
        print("  ÖRNEK 5: Güncelleme ve Silme")
        print("═" * 60)

        # ─── Güncelleme: update() ──────────────────────────────────────────
        # Sadece metadata güncelleme (embedding değişmiyor)
        koleksiyon.update(
            ids       = ["DOC006"],
            metadatas = [{"kategori": "finans", "dil": "turkce",
                          "yil": 2024, "yazar": "Uzman Analist",
                          "güvenilirlik": 0.85,  # Güncellendi: 0.75 → 0.85
                          "durum": "güncellendi"}]
        )
        print("  ✓ DOC006 metadata güncellendi (güvenilirlik: 0.75 → 0.85)")

        # ─── Upsert: Hem Metin Hem Vektör Güncelleme ──────────────────────
        yeni_metin = "Borsa yatırımı için gelişmiş teknik analiz ve risk yönetimi"
        koleksiyon.upsert(
            ids        = ["DOC006"],
            embeddings = [self.embedding.encode(yeni_metin)],
            documents  = [yeni_metin],
            metadatas  = [{"kategori": "finans", "dil": "turkce",
                           "yil": 2024, "yazar": "Uzman Analist",
                           "güvenilirlik": 0.85}]
        )
        print("  ✓ DOC006 tam güncelleme yapıldı (metin + vektör)")

        # ─── Silme: delete() ───────────────────────────────────────────────
        koleksiyon.delete(ids=["DOC011"])
        print("  ✓ DOC011 silindi")
        print(f"  📊 Kalan kayıt sayısı: {koleksiyon.count()}")

        # ─── Koşullu Silme: where ile ─────────────────────────────────────
        # Dikkat: Bazı ChromaDB sürümlerinde desteklenmeyebilir
        try:
            koleksiyon.delete(
                where={"yil": {"$lt": 2020}}  # 2020 öncesi tüm belgeler
            )
            print("  ✓ 2020 öncesi belgeler silindi")
        except Exception:
            print("  ℹ️  Koşullu silme: Bu sürümde where filtrelemeli silme")
            print("      sadece var olan ID'lerle çalışır")

    # ─────────────────────────────────────────────────────────────────────────
    # ÖRNEK 6: Gerçek Dünya — RAG Sistemi
    # ─────────────────────────────────────────────────────────────────────────

    def ornek_6_rag_sistemi(self):
        """
        RAG (Retrieval-Augmented Generation) Pattern'i.

        LLM (GPT, Claude vb.) kendi bilgisiyle cevap verir ama
        güncel/özel bilgiye erişemez. RAG bu sorunu çözer:

        ┌─────────────────────────────────────────────────────────┐
        │                    RAG Akışı                            │
        │                                                         │
        │  Kullanıcı Sorusu                                       │
        │       │                                                 │
        │       ▼                                                 │
        │  Sorgu Embedding Üret                                   │
        │       │                                                 │
        │       ▼                                                 │
        │  Vektör DB'de En Benzer Belgeleri Bul (Retrieve)        │
        │       │                                                 │
        │       ▼                                                 │
        │  Bulunan Belgeler + Soru → LLM'e Gönder (Augment)      │
        │       │                                                 │
        │       ▼                                                 │
        │  LLM Bağlam Destekli Cevap Üretir (Generate)           │
        └─────────────────────────────────────────────────────────┘
        """
        print("\n" + "═" * 60)
        print("  ÖRNEK 6: RAG (Retrieval-Augmented Generation) Sistemi")
        print("═" * 60)

        # ─── Bilgi Bankası Koleksiyonu ─────────────────────────────────────
        bilgi_bankasi = self.client.get_or_create_collection(
            name     = "sirket_bilgi_bankasi",
            metadata = {"hnsw:space": "cosine"}
        )

        # ─── Şirket Dökümanları (Chunk'lanmış) ────────────────────────────
        # Gerçekte büyük belgeler chunk'lara bölünür (örn. 512 token)
        dokumanlar = [
            ("KB001", "Ürün iade politikası: 30 gün içinde iade edilebilir. "
             "Fatura zorunludur. Online alışverişlerde kargo ücretsizdir.",
             {"kaynak": "politika", "bolum": "iade", "versiyon": "2024-v2"}),

            ("KB002", "Teknik destek saatleri: Pazartesi-Cuma 09:00-18:00. "
             "Hafta sonu 10:00-16:00. 7/24 chatbot desteği mevcuttur.",
             {"kaynak": "destek", "bolum": "saatler", "versiyon": "2024-v1"}),

            ("KB003", "Premium üyelik avantajları: %20 indirim, ücretsiz kargo, "
             "öncelikli müşteri hizmeti, özel kampanyalara erken erişim.",
             {"kaynak": "uyelik", "bolum": "premium", "versiyon": "2024-v3"}),

            ("KB004", "Ödeme yöntemleri: Kredi kartı, banka kartı, havale/EFT, "
             "kapıda ödeme (nakit/kart), taksit seçenekleri mevcuttur.",
             {"kaynak": "odeme", "bolum": "yontemler", "versiyon": "2024-v1"}),

            ("KB005", "Kargo ve teslimat: İstanbul 1-2 iş günü, diğer iller "
             "2-3 iş günü. 150 TL üzeri alışverişlerde kargo bedava.",
             {"kaynak": "kargo", "bolum": "teslimat", "versiyon": "2024-v2"}),

            ("KB006", "Garanti koşulları: Tüm elektronik ürünler 2 yıl resmi "
             "garanti kapsamındadır. Kullanıcı hatası garanti dışındadır.",
             {"kaynak": "garanti", "bolum": "kosullar", "versiyon": "2024-v1"}),
        ]

        # ─── Belgeleri Ekle ────────────────────────────────────────────────
        bilgi_bankasi.add(
            ids        = [d[0] for d in dokumanlar],
            embeddings = [self.embedding.encode(d[1]) for d in dokumanlar],
            documents  = [d[1] for d in dokumanlar],
            metadatas  = [d[2] for d in dokumanlar],
        )
        print(f"  ✓ Bilgi bankası hazır: {bilgi_bankasi.count()} döküman")

        # ─── RAG Sorguları ─────────────────────────────────────────────────
        kullanici_sorulari = [
            "Aldığım ürünü nasıl iade edebilirim?",
            "Ürünüm bozuk çıktı, garantisi var mı?",
            "Ne zaman teslim alırım, ücretsiz kargo var mı?",
        ]

        print("\n  💬 RAG Sistemi Simülasyonu")
        for soru in kullanici_sorulari:
            print(f"\n  👤 Kullanıcı: '{soru}'")

            # ─── Adım 1: Retrieve — Benzer Dökümanları Bul ────────────────
            sorgu_vec = self.embedding.encode(soru)
            bulunanlar = bilgi_bankasi.query(
                query_embeddings = [sorgu_vec],
                n_results        = 2,
                include          = ["documents", "metadatas", "distances"]
            )

            # ─── Adım 2: Augment — LLM Prompt'u Oluştur ──────────────────
            baglamlar = bulunanlar["documents"][0]
            mesafeler = bulunanlar["distances"][0]

            # Düşük güvenilirliği filtrele (threshold)
            # Cosine: distance > 0.5 ise benzerlik < 0.5, çok düşük
            gecerli_baglamlar = [
                doc for doc, dist in zip(baglamlar, mesafeler)
                if dist < 0.5  # %50'den yüksek benzerlik
            ]

            if gecerli_baglamlar:
                # Gerçekte buraya LLM API çağrısı gelir
                prompt = f"""
Sen bir müşteri hizmetleri asistanısın.
Aşağıdaki bilgilere dayanarak soruyu cevapla:

{chr(10).join(f'• {ctx}' for ctx in gecerli_baglamlar)}

Soru: {soru}"""

                print(f"  🤖 RAG Sistemi: Bağlam bulundu ({len(gecerli_baglamlar)} chunk)")
                print(f"     Kaynak: {bulunanlar['metadatas'][0][0]['kaynak']}")
                print(f"     Benzerlik: {1-mesafeler[0]:.3f}")
                # Gerçekte: response = openai.chat.completions.create(...)
                print(f"     [LLM'e gönderilecek prompt hazır — {len(prompt)} karakter]")
            else:
                print("  🤖 RAG Sistemi: İlgili bilgi bulunamadı, genel cevap verilecek")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BÖLÜM 4: MİLVUS REFERANS MİMARİSİ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MilvusReferansKodu:
    """
    Milvus için üretim seviyesi referans kod.

    KURULUM (gerçek ortam için):
      docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
      pip install pymilvus

    Milvus Mimarisi:
    ┌─────────────────────────────────────────────────────────────┐
    │                      Milvus Cluster                         │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
    │  │ Access Layer │  │ Coordinator  │  │   Worker Nodes   │  │
    │  │ (Proxy)      │  │ (Root/Query/ │  │ (QueryNode/      │  │
    │  │              │  │  Data/Index) │  │  DataNode/       │  │
    │  └──────────────┘  └──────────────┘  │  IndexNode)      │  │
    │                                      └──────────────────┘  │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │            Storage Layer                            │   │
    │  │  etcd (meta)  │  MinIO/S3 (vektörler)              │   │
    │  └─────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

    ChromaDB vs Milvus karar kriteri:
    < 1 Milyon vektör  → ChromaDB yeterli
    1M - 100M vektör   → Milvus Standalone
    > 100M vektör      → Milvus Cluster
    """

    @staticmethod
    def baglanti_ve_koleksiyon():
        """
        Milvus bağlantısı ve koleksiyon oluşturma.
        Bu kod gerçek Milvus kurulumu gerektirir.
        """
        kod = '''
# ──────────────────────────────────────────────────────────────
# Milvus: Bağlantı ve Koleksiyon Oluşturma
# pip install pymilvus
# ──────────────────────────────────────────────────────────────

from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility
)

# ─── 1. Bağlantı ──────────────────────────────────────────────
connections.connect(
    alias   = "default",
    host    = "localhost",      # Milvus sunucu adresi
    port    = "19530",          # Varsayılan port
    # user  = "root",           # Kimlik doğrulama (opsiyonel)
    # password = "Milvus",
)
print("✓ Milvus bağlantısı kuruldu")

# ─── 2. Şema Tanımlama ────────────────────────────────────────
# Milvus'da şema önceden tanımlanmalıdır (schema-first)
VEKTOR_BOYUTU = 1536  # OpenAI text-embedding-3-small boyutu

alanlar = [
    # PRIMARY KEY — otomatik ID üretimi
    FieldSchema(
        name        = "id",
        dtype       = DataType.INT64,
        is_primary  = True,
        auto_id     = True          # Milvus otomatik artıran ID üretir
    ),

    # Embedding vektörü — ana alan
    FieldSchema(
        name     = "embedding",
        dtype    = DataType.FLOAT_VECTOR,
        dim      = VEKTOR_BOYUTU    # Boyut şema tanımında sabittir!
    ),

    # Skaler alanlar — filtrelemede kullanılır
    FieldSchema(
        name     = "metin",
        dtype    = DataType.VARCHAR,
        max_length = 65535          # Maksimum karakter sayısı
    ),
    FieldSchema(
        name     = "kategori",
        dtype    = DataType.VARCHAR,
        max_length = 100
    ),
    FieldSchema(
        name     = "yil",
        dtype    = DataType.INT32
    ),
    FieldSchema(
        name     = "güvenilirlik",
        dtype    = DataType.FLOAT
    ),
    FieldSchema(
        name     = "aktif",
        dtype    = DataType.BOOL
    ),
]

sema = CollectionSchema(
    fields      = alanlar,
    description = "Şirket belgeleri vektör koleksiyonu",
    enable_dynamic_field = True    # Şemada olmayan alanları kabul et
)

# ─── 3. Koleksiyon Oluşturma ──────────────────────────────────
if utility.has_collection("belgeler"):
    utility.drop_collection("belgeler")     # Varsa sil (geliştirme)

koleksiyon = Collection(
    name        = "belgeler",
    schema      = sema,
    # shards_num: Dağıtım için shard sayısı (varsayılan: 1)
    # Büyük koleksiyonlar için artırın (önerilen: vCPU sayısı)
    shards_num  = 2,
)
print(f"✓ Koleksiyon oluşturuldu: {koleksiyon.name}")
'''
        return kod

    @staticmethod
    def indeks_ve_veri_ekleme():
        """Milvus indeks oluşturma ve veri ekleme."""
        kod = '''
# ──────────────────────────────────────────────────────────────
# Milvus: İndeks Oluşturma ve Veri Ekleme
# ──────────────────────────────────────────────────────────────

# ─── İNDEKS TÜRLERİ ───────────────────────────────────────────
#
# IVF_FLAT  → Kümeleme + tam vektör (yüksek doğruluk, yavaş inşa)
# IVF_SQ8   → IVF + Skaler Kuantizasyon (4x küçük bellek)
# IVF_PQ    → IVF + Ürün Kuantizasyon (en küçük bellek, düşük doğruluk)
# HNSW      → Graf tabanlı (hız/doğruluk dengesi, RAM yoğun)
# DISKANN   → Disk üzerinde graf (çok büyük veri, düşük bellek)
# FLAT      → Brute-force (küçük koleksiyon, %100 doğruluk)
# SCANN     → Google SCaNN (yüksek doğruluk + hız)
#
# GENEL ÖNERİ:
#   < 1M vektör:     HNSW (hız + doğruluk)
#   1M - 100M:       IVF_SQ8 (bellek verimli)
#   > 100M:          IVF_PQ + DISKANN (ölçeklenebilir)

# ─── HNSW İndeksi ──────────────────────────────────────────────
hnsw_index_params = {
    "metric_type": "COSINE",    # IP, L2, COSINE
    "index_type":  "HNSW",
    "params": {
        "M":        16,         # Graf bağlantı sayısı (4-64)
        "efConstruction": 200,  # İnşa kalitesi (8-512)
        # Yüksek M + yüksek efConstruction = iyi doğruluk, yavaş inşa
    }
}

# ─── IVF_SQ8 İndeksi (Büyük Ölçek) ───────────────────────────
ivf_index_params = {
    "metric_type": "L2",
    "index_type":  "IVF_SQ8",
    "params": {
        "nlist": 1024,   # Küme sayısı (~sqrt(N) olması önerilir)
        # 1M vektör → nlist = 1000, 100M → nlist = 10000
    }
}

# İndeksi embedding alanına uygula
koleksiyon.create_index(
    field_name   = "embedding",
    index_params = hnsw_index_params,
    index_name   = "embedding_hnsw_index"
)
print("✓ HNSW indeksi oluşturuldu")

# ─── Skaler Alan İndeksi (Filtreleme Hızlandırma) ─────────────
# Sık filtrelenen alanlar için indeks oluştur
koleksiyon.create_index(
    field_name   = "kategori",
    index_name   = "kategori_index"
)
koleksiyon.create_index(
    field_name   = "yil",
    index_name   = "yil_index"
)

# ─── Koleksiyonu Belleğe Yükle (Zorunlu!) ─────────────────────
# Milvus'da sorgu öncesi koleksiyonun yüklenmesi GEREKİR
koleksiyon.load()
print("✓ Koleksiyon belleğe yüklendi")

# ─── Veri Ekleme ──────────────────────────────────────────────
import openai

metinler = [
    "Python programlama makine öğrenmesi için idealdir",
    "Diyabet hastalığının belirtileri ve tedavisi",
    "Borsa yatırımı teknik analiz stratejileri",
]

# Gerçek OpenAI embedding (örnek)
# client = openai.OpenAI()
# response = client.embeddings.create(
#     input = metinler,
#     model = "text-embedding-3-small"
# )
# embeddingler = [item.embedding for item in response.data]

# Simüle edilmiş embedding
import random
boyut = 1536
embeddingler = [[random.gauss(0, 1) for _ in range(boyut)]
                for _ in range(len(metinler))]

# Milvus'da veri entity formatında eklenir
entities = [
    embeddingler,           # embedding alanı
    metinler,               # metin alanı
    ["teknoloji", "sağlık", "finans"],  # kategori
    [2024, 2024, 2023],     # yil
    [0.95, 0.99, 0.75],     # güvenilirlik
    [True, True, True],     # aktif
]

# field_name sırası şema sırasıyla eşleşmeli (id hariç auto)
insert_result = koleksiyon.insert(entities)
koleksiyon.flush()  # Tampon belleği diske yaz

print(f"✓ {insert_result.insert_count} kayıt eklendi")
print(f"  Eklenen ID'ler: {insert_result.primary_keys[:3]}...")
'''
        return kod

    @staticmethod
    def gelismis_sorgular():
        """Milvus gelişmiş sorgu örnekleri."""
        kod = '''
# ──────────────────────────────────────────────────────────────
# Milvus: Gelişmiş Sorgular
# ──────────────────────────────────────────────────────────────

# ─── Temel Vektör Arama ───────────────────────────────────────
sorgu_vektoru = embeddingler[0]  # Örnek sorgu vektörü

arama_params = {
    "metric_type": "COSINE",
    "params": {
        "ef": 64,        # HNSW arama derinliği (n_results'tan büyük olmalı)
        # nprobe: IVF için kaç küme taransın (1-nlist arası)
        # nprobe yüksek = doğru ama yavaş
    }
}

sonuclar = koleksiyon.search(
    data          = [sorgu_vektoru],  # Sorgu vektörleri listesi
    anns_field    = "embedding",      # Hangi vektör alanında ara
    param         = arama_params,
    limit         = 5,                # Top-K
    output_fields = ["metin", "kategori", "güvenilirlik"],
)

for hit in sonuclar[0]:
    print(f"ID: {hit.id}, Skor: {hit.score:.4f}")
    print(f"  {hit.entity.get('metin', '')[:60]}")

# ─── Filtrelemeli Arama (Hybrid) ──────────────────────────────
# Milvus filtre ifadeleri Python sözdizimi benzeri

# Basit filtre
filtre_1 = 'kategori == "teknoloji"'

# Karmaşık filtre
filtre_2 = (
    'kategori in ["teknoloji", "saglik"]'
    ' AND guvenilirlik > 0.85'
    ' AND yil >= 2023'
    ' AND aktif == True'
)

# Range filtre
filtre_3 = '50 <= yil < 2025'

sonuclar_filtreli = koleksiyon.search(
    data          = [sorgu_vektoru],
    anns_field    = "embedding",
    param         = arama_params,
    limit         = 10,
    expr          = filtre_2,          # Boolean filtre ifadesi
    output_fields = ["metin", "kategori", "güvenilirlik", "yil"],
)

# ─── Batch Search — Çoklu Sorgu ───────────────────────────────
# Birden fazla sorguyu paralel çalıştır (verimli!)
coklu_sonuclar = koleksiyon.search(
    data  = embeddingler[:3],   # 3 sorgu aynı anda
    anns_field = "embedding",
    param      = arama_params,
    limit      = 5,
)
# coklu_sonuclar[0] → 1. sorgu sonuçları
# coklu_sonuclar[1] → 2. sorgu sonuçları
# coklu_sonuclar[2] → 3. sorgu sonuçları

# ─── Query — Sadece Metadata Filtre (Vektörsüz) ───────────────
# Geleneksel SQL SELECT gibi çalışır
skaler_sonuclar = koleksiyon.query(
    expr          = 'kategori == "sağlık" AND güvenilirlik > 0.90',
    output_fields = ["id", "metin", "güvenilirlik"],
    limit         = 100,
    offset        = 0,     # Sayfalama için
)
print(f"Sağlık kategorisinde güvenilir: {len(skaler_sonuclar)} kayıt")

# ─── Partition — Büyük Koleksiyon Optimizasyonu ───────────────
# Veriyi mantıksal bölmelere ayırma (performans için)
koleksiyon.create_partition("teknoloji_2024")
koleksiyon.create_partition("saglik_2024")

# Belirli partition'da arama (daha hızlı!)
bolumlu_sonuc = koleksiyon.search(
    data             = [sorgu_vektoru],
    anns_field       = "embedding",
    param            = arama_params,
    limit            = 5,
    partition_names  = ["teknoloji_2024"],  # Sadece bu bölümde ara
)

# ─── Delete — Vektör Silme ────────────────────────────────────
# ID'ye göre silme
koleksiyon.delete(expr="id in [1, 2, 3]")

# Koşula göre silme
koleksiyon.delete(expr='kategori == "test" AND aktif == False')

# ─── Koleksiyon Yönetimi ──────────────────────────────────────
# Bellek'ten kaldır (kaynakları serbest bırak)
koleksiyon.release()

# Veri sayısı
print(f"Toplam kayıt: {koleksiyon.num_entities}")

# Koleksiyon bilgisi
print(koleksiyon.describe())

# İstatistikler
stats = utility.get_query_segment_info("belgeler")
'''
        return kod


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BÖLÜM 5: PERFORMANS VE OPTİMİZASYON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PerformansAnalizi:
    """
    Vektör veritabanı performans metrikleri ve optimizasyon ipuçları.
    """

    def __init__(self):
        self.embedding = EmbeddingMotoru(boyut=64)
        self.matematik = VektorMatematigi()

    def benchmark_calistir(self):
        """
        ChromaDB üzerinde temel performans benchmarkı.
        Farklı veri boyutlarında sorgu sürelerini ölçer.
        """
        print("\n" + "═" * 60)
        print("  PERFORMANS ANALİZİ & BENCHMARK")
        print("═" * 60)

        # ─── Test Verisi Üret ──────────────────────────────────────────────
        print("\n  📊 Test koleksiyonu oluşturuluyor...")
        test_client = chromadb.EphemeralClient()
        test_kol = test_client.create_collection(
            name="benchmark",
            metadata={"hnsw:space": "cosine"}
        )

        # 1000 rastgele vektör ekle
        N = 1000
        ids = [f"BENCH_{i:05d}" for i in range(N)]
        kategoriler = ["teknoloji", "sağlık", "finans", "spor", "yemek"]
        metinler = [f"örnek metin {i} kategori test verisi" for i in range(N)]
        embeddingler = [self.embedding.encode(m) for m in metinler]
        metadatalar = [
            {"kategori": kategoriler[i % len(kategoriler)],
             "index": i, "grup": i // 100}
            for i in range(N)
        ]

        ekle_baslangic = time.perf_counter()
        test_kol.add(
            ids=ids, embeddings=embeddingler,
            documents=metinler, metadatas=metadatalar
        )
        ekle_sure = time.perf_counter() - ekle_baslangic
        print(f"  ✓ {N} kayıt eklendi: {ekle_sure:.3f}s "
              f"({N/ekle_sure:.0f} kayıt/sn)")

        # ─── Sorgu Benchmark'ı ─────────────────────────────────────────────
        print("\n  ⚡ Sorgu Performans Testi:")
        print(f"  {'─'*50}")
        print(f"  {'Test':<35} {'Süre (ms)':<12} {'Sonuç'}")
        print(f"  {'─'*50}")

        testler = [
            ("Basit sorgu (n=1)",
             lambda: test_kol.query(
                 query_embeddings=[self.embedding.encode("test sorgu")],
                 n_results=1, include=["documents"])),
            ("Top-10 sorgu",
             lambda: test_kol.query(
                 query_embeddings=[self.embedding.encode("test sorgu")],
                 n_results=10, include=["documents", "distances"])),
            ("Metadata filtrelemeli sorgu",
             lambda: test_kol.query(
                 query_embeddings=[self.embedding.encode("teknoloji")],
                 n_results=5,
                 where={"kategori": {"$eq": "teknoloji"}},
                 include=["documents", "distances"])),
            ("Batch sorgu (5 sorgu)",
             lambda: test_kol.query(
                 query_embeddings=[self.embedding.encode(f"sorgu {i}")
                                   for i in range(5)],
                 n_results=3, include=["documents"])),
            ("ID ile get()",
             lambda: test_kol.get(
                 ids=[f"BENCH_{i:05d}" for i in range(10)],
                 include=["documents", "metadatas"])),
        ]

        for ad, test_fn in testler:
            # Her testi 5 kez çalıştır, ortalamasını al
            sureler = []
            for _ in range(5):
                baslangic = time.perf_counter()
                sonuc = test_fn()
                sure = (time.perf_counter() - baslangic) * 1000
                sureler.append(sure)

            ortalama = sum(sureler) / len(sureler)
            minimum = min(sureler)

            # Sonuç sayısı
            if hasattr(sonuc, '__getitem__') and 'documents' in sonuc:
                docs = sonuc['documents']
                if isinstance(docs[0], list):
                    sayi = sum(len(d) for d in docs)
                else:
                    sayi = len(docs)
            else:
                sayi = "?"

            print(f"  {ad:<35} {ortalama:>6.2f}ms     {sayi}")

        # ─── Optimizasyon İpuçları ────────────────────────────────────────
        print(f"\n  {'═'*58}")
        print("  💡 PERFORMANS OPTİMİZASYON İPUÇLARI")
        print(f"  {'═'*58}")

        ipuclari = [
            ("Embedding Önbelleği",
             "Aynı metni tekrar encode etmeyin → LRU cache kullanın"),
            ("Batch Ekleme",
             "Tek tek add() yerine toplu add() → 10-50x hız kazanımı"),
            ("include Alanlarını Sınırla",
             "Embeddings dahil etmeyin (büyük veri), sadece ihtiyacı isteyin"),
            ("Metadata İndeksi",
             "Sık filtrelenen alanlar için indeks oluşturun"),
            ("Vektör Boyutu",
             "1536 boyut yerine 384 boyut modeli → 4x daha hızlı"),
            ("Quantization",
             "float32 → int8 quantization → 4x küçük bellek, %5 doğruluk kaybı"),
            ("Sayfalama",
             "n_results büyük tutmak yerine offset/limit ile sayfalayın"),
            ("Persistent vs Ephemeral",
             "Prod: PersistentClient, Test: EphemeralClient"),
        ]

        for baslik, aciklama in ipuclari:
            print(f"  • {baslik:<25} → {aciklama}")

    def benzerlik_karsilastirma(self):
        """
        Farklı cümlelerin benzerlik skorlarını görselleştirir.
        """
        print("\n" + "═" * 60)
        print("  BENZERLİK SKORU ANALİZİ")
        print("═" * 60)

        referans = "Python ile veri bilimi ve makine öğrenmesi"
        karsilastirmalar = [
            ("Python yapay zeka derin öğrenme", "Çok Yakın"),
            ("Makine öğrenmesi algoritmaları kod", "Yakın"),
            ("Programlama dillerinin karşılaştırması", "Orta"),
            ("Futbol maçı sonuçları haftalık özet", "Uzak"),
            ("Pizza tarifi malzemeleri ve pişirme süresi", "Çok Uzak"),
        ]

        ref_vec = self.embedding.encode(referans)
        print(f"\n  Referans: '{referans}'")
        print(f"  {'─'*58}")
        print(f"  {'Metin':<45} {'Benzerlik':>8}  Beklenti")
        print(f"  {'─'*58}")

        for metin, beklenti in karsilastirmalar:
            vec = self.embedding.encode(metin)
            skor = self.matematik.kosinüs_benzerligi(ref_vec, vec)
            bar = "█" * int(max(0, skor) * 20) + "░" * (20 - int(max(0, skor) * 20))
            print(f"  {metin:<45} {skor:>8.4f}  {beklenti}")
            print(f"  {' '*45} [{bar}]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BÖLÜM 6: MİMARİ KARŞILAŞTIRMA ÖZET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mimari_ozet():
    """Genel mimari karşılaştırma ve karar rehberi."""
    print("\n" + "═" * 65)
    print("  MİMARİ KARŞILAŞTIRMA & KARAR REHBERİ")
    print("═" * 65)

    tablo = """
  ┌──────────────┬──────────┬──────────┬─────────┬──────────────┐
  │ Özellik      │ ChromaDB │  Milvus  │Pinecone │  pgvector    │
  ├──────────────┼──────────┼──────────┼─────────┼──────────────┤
  │ Kurulum      │  pip     │  Docker  │ Yok     │ PostgreSQL   │
  │ Ölçek        │  ≤10M    │  ≤100B   │ Yönetimli│ ≤50M        │
  │ Hız (sorgu)  │  ms      │  μs-ms   │  ms     │  ms          │
  │ Maliyet      │  Ücretsiz│ Açık Kay.│ Ücretli │  Ücretsiz    │
  │ SQL Desteği  │  Yok     │  Yok     │ Yok     │  Tam SQL     │
  │ Cluster      │  Yok     │  Var     │ Var     │  PG Cluster  │
  │ Güncellik    │  Yüksek  │  Yüksek  │ Orta    │  Düşük       │
  └──────────────┴──────────┴──────────┴─────────┴──────────────┘

  KARAR AĞACI:
  ────────────
  SQL + Vektör birlikte mi?        → pgvector
  Hızlı prototip / RAG uygulaması? → ChromaDB
  Üretim, büyük ölçek?             → Milvus
  Sıfır operasyon yükü?            → Pinecone / Weaviate Cloud
  Rust/Go performansı?             → Qdrant
  GraphQL + semantik arama?        → Weaviate

  EMBEDDING MODELİ SEÇİMİ:
  ──────────────────────────
  Türkçe destekli:    paraphrase-multilingual-MiniLM-L12-v2
  Hız odaklı:         all-MiniLM-L6-v2         (384 boyut)
  Doğruluk odaklı:    text-embedding-3-large   (3072 boyut)
  Ücretsiz + güçlü:   BAAI/bge-large-en-v1.5   (1024 boyut)
  GPT entegrasyonu:   text-embedding-3-small   (1536 boyut)
"""
    print(tablo)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANA ÇALIŞMA BLOĞU
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("█" * 65)
    print("  VEKTÖR VERİTABANLARI — Kapsamlı Rehber")
    print("  ChromaDB · Milvus · Embedding · ANN Arama")
    print("█" * 65)

    # ─── BÖLÜM 1: Matematik Temelleri ─────────────────────────────────────
    print("\n🔢 VEKTÖR MATEMATİĞİ")
    print("─" * 40)
    mat = VektorMatematigi()
    a = [1.0, 0.0, 0.5, -0.3]
    b = [0.9, 0.1, 0.4, -0.2]
    c = [-1.0, 0.0, -0.5, 0.3]

    print(f"  a = {a}")
    print(f"  b = {b} (a'ya benzer)")
    print(f"  c = {c} (a'nın zıttı)")
    print(f"\n  Kosinüs (a, b) = {mat.kosinüs_benzerligi(a, b):.4f} ← yakın")
    print(f"  Kosinüs (a, c) = {mat.kosinüs_benzerligi(a, c):.4f} ← zıt")
    print(f"  Öklid   (a, b) = {mat.oklid_mesafesi(a, b):.4f} ← küçük mesafe")
    print(f"  Öklid   (a, c) = {mat.oklid_mesafesi(a, c):.4f} ← büyük mesafe")

    # ─── BÖLÜM 2: Embedding ───────────────────────────────────────────────
    print("\n\n📐 EMBEDDİNG MOTEURü")
    print("─" * 40)
    motor = EmbeddingMotoru(boyut=64)
    test_metinler = [
        "Python programlama",
        "Yapay zeka derin öğrenme",
        "Futbol maçı sonuçları",
    ]
    for metin in test_metinler:
        vec = motor.encode(metin)
        print(f"  '{metin}'")
        print(f"   Vektör boyutu: {len(vec)}, "
              f"İlk 5 değer: {[f'{v:.3f}' for v in vec[:5]]}")

    # ─── BÖLÜM 3: ChromaDB Örnekleri ──────────────────────────────────────
    print("\n\n🗄️  CHROMADB ÖRNEKLERİ")
    chroma = ChromaDBOrnek()
    kol = chroma.ornek_1_koleksiyon_yonetimi()
    belgeler = chroma.ornek_2_veri_ekleme(kol)
    chroma.ornek_3_semantik_arama(kol)
    chroma.ornek_4_metadata_filtreleme(kol)
    chroma.ornek_5_guncelleme_silme(kol)
    chroma.ornek_6_rag_sistemi()

    # ─── BÖLÜM 4: Milvus Referans Kodu ────────────────────────────────────
    print("\n\n🚀 MİLVUS REFERANS KODU")
    print("═" * 60)
    print("  Milvus gerçek kurulum gerektirdiğinden kod referansı gösteriliyor.")
    print("  Kurulum: docker run -d -p 19530:19530 milvusdb/milvus:latest")
    print("  Kurulum: pip install pymilvus")
    milvus = MilvusReferansKodu()
    for baslik, fonk in [
        ("Bağlantı ve Koleksiyon", milvus.baglanti_ve_koleksiyon),
        ("İndeks ve Veri Ekleme", milvus.indeks_ve_veri_ekleme),
        ("Gelişmiş Sorgular", milvus.gelismis_sorgular),
    ]:
        print(f"\n  📌 {baslik}")
        print(f"  {'─' * 55}")
        # Kodu göster
        for satir in fonk().strip().split('\n')[:8]:
            print(f"  {satir}")
        print("  ... (devam ediyor)")

    # ─── BÖLÜM 5: Performans Analizi ──────────────────────────────────────
    perf = PerformansAnalizi()
    perf.benzerlik_karsilastirma()
    perf.benchmark_calistir()

    # ─── BÖLÜM 6: Mimari Özet ─────────────────────────────────────────────
    mimari_ozet()

    print("\n" + "█" * 65)
    print("  ✅ Tüm örnekler başarıyla çalıştırıldı!")
    print("█" * 65 + "\n")
