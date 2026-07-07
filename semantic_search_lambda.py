"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      SEMANTİK ARAMA — Yerelde Gerçeğe Yakın Simülasyon Rehberi               ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SEMANTİK ARAMA NEDİR?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Klasik Arama (BM25/TF-IDF):
    Sorgu: "araba tamiri"
    Bulur: "araba tamiri" kelimesini içeren belgeler
    Bulamaz: "otomobil onarımı", "araç servisi" ← Aynı anlam, farklı kelime!

  Semantik Arama:
    Sorgu: "araba tamiri"
    Bulur: "otomobil onarımı" ✓
           "araç bakım servisi" ✓
           "motor arızası giderilmesi" ✓
           → Kelime değil ANLAM benzerliği arar

  ┌──────────────────────────────────────────────────────────────┐
  │                   Semantik Arama Akışı                        │
  │                                                               │
  │  İNDEKSLEME (Tek Seferlik):                                   │
  │  Belgeler → Embedding Modeli → Vektörler → Vektör DB          │
  │                                                               │
  │  ARAMA (Her Sorguda):                                         │
  │  Sorgu → Embedding Modeli → Sorgu Vektörü                     │
  │       → Vektör DB'de En Yakın K Belge (ANN)                   │
  │       → [Opsiyonel] Reranking ile Sıralama Düzeltme           │
  │       → Sonuçlar                                              │
  └──────────────────────────────────────────────────────────────┘

2. YEREL SİMÜLASYON NEDİR?
━━━━━━━━━━━━━━━━━━━━━━━━━━
    Bu dosya, dış LLM/Embedding servislerine bağlı kalmadan semantik arama
    davranışını yerelde gerçekçi şekilde simüle eder.

    Neden yerel simülasyon?
    • API anahtarı gerekmez
    • İnternetsiz/izole ortamlarda çalışır
    • Deterministik sonuç üretir (aynı girdi → aynı çıktı)
    • Hızlı geliştirme ve test döngüsü sağlar

    Simülasyon prensibi:
    • Metinler alan-tabanlı + hash + n-gram ile vektörlenir
    • Vektörler normalize edilip kosinüs benzerliğiyle aranır
    • RAG cevabı kural tabanlı/şablon tabanlı üretilir

3. SEMANTİK ARAMA MİMARİLERİ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Tier 1 — Saf Semantik Arama:
    Sorgu → Embedding → Vektör ANN → Sonuçlar
    Kullanım: Basit doküman arama, Q&A

  Tier 2 — Hibrit Arama (Bu dosyada):
    Sorgu → [Semantik Skor + BM25 Skor] → Fusion → Rerank → Sonuçlar
    Kullanım: E-ticaret, haber, akademik arama

  Tier 3 — RAG Pipeline:
    Sorgu → Hibrit Arama → Bağlam Seçimi → LLM → Cevap
    Kullanım: Chatbot, döküman asistanı, bilgi bankası
"""

# ─── Standart Kütüphaneler ────────────────────────────────────────────────
import os
import re
import time
import json
import math
import hashlib
import logging
import asyncio
import unittest
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

# ─── Üçüncü Taraf Kütüphaneler ───────────────────────────────────────────
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi          # Klasik BM25 tam metin araması

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("semantic_search")


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: YEREL AI İSTEMCİSİ
# ══════════════════════════════════════════════════════════════════════════════

class YerelAIIstemcisi:
    """
    Yerel (offline-first) embedding üretimi ve LLM benzeri metin üretimi.

    Varsayılan davranış: Tamamen yerel simülasyon.
    İsteğe bağlı olarak dış bir OpenAI-uyumlu endpoint'e geçiş mümkündür,
    fakat sadece explicit olarak izin verilirse kullanılır.

    ─── Gerçek Kullanım (API key varsa) ────────────────────────────────────

    Yöntem A — openai paketi ile (önerilen):
        pip install openai

        from openai import OpenAI
        client = OpenAI(
            api_key  = "service-key-xxxx",
            base_url = "https://example.com/v1"
        )
        response = client.embeddings.create(
            model = "text-embedding-ada-002",
            input = ["Merhaba dünya", "Python programlama"]
        )
        vektorler = [item.embedding for item in response.data]

    Yöntem B — requests ile (bağımlılıksız):
        import requests
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "text-embedding-ada-002",
            "input": ["metin 1", "metin 2"]
        }
        r = requests.post(
            "https://example.com/v1/embeddings",
            headers=headers, json=payload
        )
        vektorler = [item["embedding"] for item in r.json()["data"]]

    Yöntem C — LLM için (chat completions):
        response = client.chat.completions.create(
            model = "local-sim-chat-v1",
            messages = [
                {"role": "system", "content": "Sen bir asistansın"},
                {"role": "user",   "content": "Merhaba!"}
            ],
            temperature = 0.7,
            max_tokens  = 512
        )
        cevap = response.choices[0].message.content

    ─── Bu Dosyada: Yerel Simülasyon Modu (Varsayılan) ─────────────────────
    API key gerektirmeden deterministik embedding üretir.
    Benzer metinler yakın, farklı metinler uzak vektörler alır.
    """

    # İsteğe bağlı dış servis ayarları (varsayılan: kapalı)
    BASE_URL     = os.getenv("EMBEDDING_API_BASE_URL", "")
    EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "local-sim-embed-v1")
    CHAT_MODEL   = os.getenv("CHAT_MODEL", "local-sim-chat-v1")
    EMBED_DIM    = 128   # Gerçekte 1536; demo için 128

    # Semantik alan grupları: benzer anlam → benzer vektör bölgesi
    _ALANI: Dict[str, List[str]] = {
        "teknoloji":  ["python", "javascript", "kod", "yazılım", "programlama",
                       "algoritma", "api", "veritabanı", "geliştirici", "framework",
                       "yapay zeka", "makine öğrenmesi", "deep learning", "neural",
                       "model", "veri", "analiz", "cloud", "docker", "kubernetes"],
        "sağlık":     ["hastalık", "tedavi", "doktor", "hastane", "ilaç",
                       "sağlık", "tıp", "hasta", "klinik", "tanı", "semptom",
                       "beslenme", "diyet", "vitamin", "egzersiz", "spor"],
        "finans":     ["para", "yatırım", "borsa", "hisse", "kripto", "bitcoin",
                       "ekonomi", "faiz", "döviz", "banka", "finans", "fiyat",
                       "maliyet", "bütçe", "gelir", "gider", "vergi"],
        "hukuk":      ["kanun", "mahkeme", "dava", "avukat", "hukuk", "yasa",
                       "sözleşme", "hak", "ceza", "suç", "adalet", "karar"],
        "bilim":      ["araştırma", "deney", "hipotez", "teori", "kimya",
                       "fizik", "biyoloji", "matematik", "formül", "ispat"],
        "eğlence":    ["film", "müzik", "oyun", "roman", "kitap", "sanat",
                       "tiyatro", "sinema", "konser", "dizi", "podcast"],
    }

    def __init__(self,
                 api_key: Optional[str] = None,
                 simule: bool = True,
                 uzak_servis_izinli: bool = False):
        """
        api_key            : Opsiyonel dış servis API anahtarı
        simule             : True ise her durumda yerel simülasyon kullan
        uzak_servis_izinli : True ise dış API kullanımı denenebilir
        """
        self.api_key = api_key or os.getenv("LOCAL_AI_API_KEY", "")
        self.uzak_servis_izinli = uzak_servis_izinli
        self.simule = simule or not (self.uzak_servis_izinli and self.api_key and self.BASE_URL)
        self._cache: Dict[str, List[float]] = {}
        self._api_cagrisi = 0    # API çağrı sayacı (maliyet takibi)

        mod = "Yerel Simülasyon" if self.simule else "Dış API"
        log.info(f"YerelAIIstemcisi başlatıldı [{mod}] dim={self.EMBED_DIM}")

    # ─── Embedding Üretimi ────────────────────────────────────────────────

    def embedding_uret(self, metinler: List[str]) -> List[List[float]]:
        """
        Metin listesini vektör listesine dönüştürür.

        Gerçek API çağrısı:
            POST /v1/embeddings
            {"model": "text-embedding-ada-002", "input": [...]}

        Dönen yapı:
            {"data": [{"embedding": [...], "index": 0}, ...]}
        """
        if not self.simule and self.api_key:
            return self._gercek_api_embedding(metinler)
        return [self._simule_embedding(m) for m in metinler]

    def _gercek_api_embedding(self, metinler: List[str]) -> List[List[float]]:
        """
        Gerçek dış API çağrısı.
        Bu metod yalnızca geçerli API key olduğunda çağrılır.
        """
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.EMBED_MODEL,
            "input": metinler
        }
        try:
            r = requests.post(
                f"{self.BASE_URL}/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )
            r.raise_for_status()
            self._api_cagrisi += 1
            veri = r.json()
            # Sıralama garantisi için index'e göre sırala
            veri["data"].sort(key=lambda x: x["index"])
            return [item["embedding"] for item in veri["data"]]
        except Exception as e:
            log.warning(f"Dış API hatası, yerel simülasyona düşülüyor: {e}")
            return [self._simule_embedding(m) for m in metinler]

    def _simule_embedding(self, metin: str) -> List[float]:
        """
        API olmadan deterministik vektör üretir.

        Üç katmanlı kodlama:
        1. Semantik alan ağırlıkları (anlam grupları)
        2. Hash tabanlı benzersizlik (farklı metinler farklı vektör)
        3. N-gram özelliği (alt kelime bilgisi)
        """
        if metin in self._cache:
            return self._cache[metin]

        v = np.zeros(self.EMBED_DIM)
        m = metin.lower()

        # Katman 1: Semantik alan kodlaması
        alan_sayisi = len(self._ALANI)
        bolum = self.EMBED_DIM // alan_sayisi
        for i, (alan, kelimeler) in enumerate(self._ALANI.items()):
            skor = sum(1.5 if k == m else (1.0 if k in m else 0)
                       for k in kelimeler)
            if skor > 0:
                bas, bit = i * bolum, (i + 1) * bolum
                alan_v = np.array([
                    skor * math.sin(j * 0.8 + i * 1.3)
                    for j in range(bit - bas)
                ])
                v[bas:bit] += alan_v

        # Katman 2: Hash gürültüsü (benzersizlik)
        h = hashlib.sha256(metin.encode()).digest()
        # digest() sadece 32 byte; EMBED_DIM kadar olması için döngüyle doldur
        hash_bytes = (list(h) * (self.EMBED_DIM // len(h) + 1))[:self.EMBED_DIM]
        noise = np.array([(b / 127.5) - 1.0 for b in hash_bytes])
        v += noise * 0.2

        # Katman 3: Bigram özellikleri
        for i in range(len(m) - 1):
            bg = m[i:i+2]
            idx = int(hashlib.md5(bg.encode()).hexdigest()[:4], 16) % self.EMBED_DIM
            v[idx] += 0.05

        # L2 normalizasyon (kosinüs benzerliği için)
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm

        sonuc = v.tolist()
        self._cache[metin] = sonuc
        return sonuc

    def llm_cevap_uret(self, sistem: str, kullanici: str,
                       sicaklik: float = 0.3) -> str:
        """
        Yerel simülasyon veya opsiyonel dış API ile metin üretimi.

        Gerçek API:
            POST /v1/chat/completions
            {
              "model": "llama-4-maverick-17b-128e-instruct-fp8",
              "messages": [
                  {"role": "system", "content": sistem},
                  {"role": "user",   "content": kullanici}
              ],
              "temperature": 0.3
            }
        """
        if not self.simule and self.api_key:
            import requests
            r = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": self.CHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": sistem},
                        {"role": "user",   "content": kullanici}
                    ],
                    "temperature": sicaklik,
                    "max_tokens": 1024
                },
                timeout=60
            )
            return r.json()["choices"][0]["message"]["content"]

        # Simülasyon: kural tabanlı cevap
        return f"[Yerel Simülasyon] '{kullanici[:50]}...' sorusuna yerel LLM benzeri cevap."

    def maliyet_raporu(self) -> Dict[str, Any]:
        """API kullanım maliyet tahmini (dış servis aktifse)."""
        # Genel tahmin (örnek değer)
        token_basi_usd = 0.0000001  # $0.10 / 1M token
        toplam_token   = self._api_cagrisi * 512  # Ortalama tahmin
        return {
            "api_cagrisi"    : self._api_cagrisi,
            "tahmini_token"  : toplam_token,
            "tahmini_maliyet": f"${toplam_token * token_basi_usd:.6f}",
            "onbellekte"     : len(self._cache),
        }


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: VERİ YAPILARI
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Belge:
    """
    Arama sistemindeki tek bir belge birimi.

    Gerçek uygulamada bu sınıf bir veritabanı ORM modeliyle
    veya Pydantic şemasıyla değiştirilebilir.
    """
    id          : str
    baslik      : str
    icerik      : str
    kategori    : str
    yazar       : str
    tarih       : str
    etiketler   : List[str]   = field(default_factory=list)
    puan        : float       = 0.0          # Ortalama kullanıcı puanı
    goruntuleme : int         = 0
    embedding   : List[float] = field(default_factory=list, repr=False)

    @property
    def tam_metin(self) -> str:
        """Başlık + içerik + etiketlerin birleşimi (indeksleme için)."""
        return f"{self.baslik}. {self.icerik} {' '.join(self.etiketler)}"

    def ozet(self, uzunluk: int = 120) -> str:
        """Görüntüleme için kısaltılmış içerik."""
        return self.icerik[:uzunluk] + "..." if len(self.icerik) > uzunluk else self.icerik


@dataclass
class AramaSonucu:
    """
    Bir arama sorgusunun tek bir sonucu.
    Birden fazla skor kaynağını bir araya getirir.
    """
    belge           : Belge
    semantik_skor   : float = 0.0   # Kosinüs benzerliği (0-1)
    bm25_skor       : float = 0.0   # BM25 tam metin skoru (0-∞)
    rerank_skor     : float = 0.0   # Cross-encoder rerank skoru (0-1)
    nihai_skor      : float = 0.0   # Fusion sonrası birleşik skor
    eslesen_alan    : str   = ""    # Hangi alanda eşleşti
    aciklama        : str   = ""    # Neden bu sonuç seçildi


@dataclass
class AramaIstatistigi:
    """Her arama isteğinin performans ölçümlerini tutar."""
    sorgu           : str
    bulunan_sonuc   : int
    semantik_ms     : float = 0.0
    bm25_ms         : float = 0.0
    rerank_ms       : float = 0.0
    toplam_ms       : float = 0.0
    strateji        : str   = ""


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: BM25 TAM METİN ARAMA MOTORU
# ══════════════════════════════════════════════════════════════════════════════

class BM25Motor:
    """
    BM25 (Best Match 25) — Klasik tam metin arama algoritması.

    Wikipedia, Elasticsearch ve Solr'ın temel algoritması budur.

    BM25 Formülü:
        score(D, Q) = Σ IDF(qᵢ) × [tf(qᵢ,D) × (k₁+1)] / [tf(qᵢ,D) + k₁ × (1 - b + b×|D|/avgdl)]

    Parametreler:
        k₁ = 1.5  : Terim frekansı doyumu (yüksek = frekans önemli)
        b  = 0.75 : Uzunluk normalizasyonu (1.0 = tam normaliz)

    Ne zaman kullanılır?
    • Tam kelime eşleştirme gerektiğinde (model adı, ürün kodu)
    • Kısa sorgularda (1-3 kelime)
    • Anlam benzerliği gerekmediğinde

    Ne zaman yetersiz kalır?
    • Eş anlamlılar: "araba" ≠ "otomobil"
    • Farklı dil/yazım: "ML" ≠ "machine learning"
    • Kavramsal sorgular: "en iyi Python kitabı" ← anlam gerektirir
    """

    def __init__(self):
        self.bm25_modeli: Optional[BM25Okapi] = None
        self.belgeler: List[Belge] = []
        self._tokenler: List[List[str]] = []

    def _tokenize(self, metin: str) -> List[str]:
        """
        Metni BM25 için token listesine dönüştürür.

        Basit tokenizer: küçük harf + noktalama temizleme.
        Üretimde: Zemberek (Türkçe NLP) veya NLTK kullanın.
        """
        metin = metin.lower()
        # Noktalama işaretlerini kaldır
        metin = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ]', ' ', metin)
        # Birden fazla boşluğu tek boşluğa indir
        tokenler = metin.split()
        # 2 karakterden kısa kelimeleri at (Türkçe stop word benzeri)
        return [t for t in tokenler if len(t) > 2]

    def indeksle(self, belgeler: List[Belge]) -> None:
        """Belge koleksiyonundan BM25 indeksi oluşturur."""
        self.belgeler = belgeler
        self._tokenler = [self._tokenize(b.tam_metin) for b in belgeler]
        self.bm25_modeli = BM25Okapi(
            self._tokenler,
            k1=1.5,   # Terim frekansı doyumu
            b=0.75    # Uzunluk normalizasyonu
        )
        log.info(f"BM25 indeksi oluşturuldu: {len(belgeler)} belge")

    def ara(self, sorgu: str, k: int = 20) -> List[Tuple[Belge, float]]:
        """
        BM25 ile tam metin araması.

        Returns: [(belge, skor), ...] azalan skor sırasıyla
        """
        if not self.bm25_modeli:
            return []
        sorgu_tokenleri = self._tokenize(sorgu)
        if not sorgu_tokenleri:
            return []
        skorlar = self.bm25_modeli.get_scores(sorgu_tokenleri)
        # Skor > 0 olanları sırala ve ilk k tanesini döndür
        indeksler = np.argsort(skorlar)[::-1]
        sonuclar = [
            (self.belgeler[i], float(skorlar[i]))
            for i in indeksler[:k]
            if skorlar[i] > 0
        ]
        return sonuclar


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: SEMANTİK ARAMA MOTORU
# ══════════════════════════════════════════════════════════════════════════════

class SemanticAramaMotoru:
    """
    ChromaDB + YerelAIIstemcisi embeddings ile vektör tabanlı arama.

    Vektör indeksi: HNSW (Hierarchical Navigable Small World)
    • Büyük koleksiyonlarda O(log N) arama karmaşıklığı
    • %95+ doğruluk, brute-force'tan 100x+ hızlı
    • Bellek: ~4 byte × boyut × belge_sayısı

    HNSW parametreleri (ChromaDB varsayılanları):
        M=16         : Her düğümün bağlantı sayısı (4-64)
        ef_construct : İnşa sırasında komşu arama genişliği
        ef_search    : Sorgu sırasında komşu arama genişliği
    """

    def __init__(self, lambda_istemcisi: YerelAIIstemcisi,
                 koleksiyon_adi: str = "semantic_search_demo"):
        self.lambda_api = lambda_istemcisi
        self.db_istemcisi = chromadb.EphemeralClient()

        # HNSW ile ChromaDB koleksiyonu (cosine mesafesi)
        self.koleksiyon = self.db_istemcisi.create_collection(
            name=koleksiyon_adi,
            metadata={
                "hnsw:space"           : "cosine",
                "hnsw:M"               : 16,
                "hnsw:construction_ef" : 200,
                "hnsw:search_ef"       : 100,
            }
        )
        self._belgeler: Dict[str, Belge] = {}
        log.info(f"Semantik motor hazır: koleksiyon='{koleksiyon_adi}'")

    def indeksle(self, belgeler: List[Belge], batch_boyutu: int = 50) -> None:
        """
        Belgeleri embedding vektörlerine dönüştürüp ChromaDB'ye ekler.

        batch_boyutu: API sınırlı olduğunda kaçar kaçar işleneceği.
        Yerel modda performans için 50-200 arası batch genelde uygundur.
        """
        log.info(f"{len(belgeler)} belge indeksleniyor...")
        baslangic = time.perf_counter()

        for i in range(0, len(belgeler), batch_boyutu):
            batch = belgeler[i : i + batch_boyutu]
            metinler = [b.tam_metin for b in batch]

            # Yerel istemciden embedding üret
            embeddingler = self.lambda_api.embedding_uret(metinler)

            # Embedding'i belge nesnelerine ata (arama sonucunda kullanılır)
            for b, emb in zip(batch, embeddingler):
                b.embedding = emb
                self._belgeler[b.id] = b

            # ChromaDB'ye ekle
            self.koleksiyon.add(
                ids        = [b.id for b in batch],
                embeddings = embeddingler,
                documents  = [b.tam_metin for b in batch],
                metadatas  = [{
                    "baslik"    : b.baslik,
                    "kategori"  : b.kategori,
                    "yazar"     : b.yazar,
                    "tarih"     : b.tarih,
                    "puan"      : b.puan,
                    "etiketler" : ",".join(b.etiketler),
                } for b in batch]
            )

        sure = (time.perf_counter() - baslangic) * 1000
        log.info(f"İndeksleme tamamlandı: {len(belgeler)} belge, {sure:.1f}ms")

    def ara(self, sorgu: str, k: int = 10,
            filtreler: Optional[Dict] = None) -> List[Tuple[Belge, float]]:
        """
        Semantik benzerlik araması.

        sorgu   : Doğal dil sorgusu
        k       : Kaç sonuç dönsün
        filtreler: ChromaDB where filtresi (metadata üzerinde)

        Returns: [(belge, benzerlik_skoru), ...] — 0 ile 1 arasında
        """
        # Sorguyu vektöre dönüştür
        sorgu_vektoru = self.lambda_api.embedding_uret([sorgu])[0]

        # ChromaDB'de yaklaşık K-NN araması
        sorgu_kwargs: Dict[str, Any] = {
            "query_embeddings": [sorgu_vektoru],
            "n_results"       : k,
            "include"         : ["documents", "metadatas", "distances"],
        }
        if filtreler:
            sorgu_kwargs["where"] = filtreler

        sonuclar = self.koleksiyon.query(**sorgu_kwargs)

        # Sonuçları döndür
        cikti = []
        for belge_id, meta, uzaklik in zip(
            sonuclar["ids"][0],
            sonuclar["metadatas"][0],
            sonuclar["distances"][0]
        ):
            if belge_id in self._belgeler:
                # ChromaDB cosine distance → similarity dönüşümü
                # distance ∈ [0,2], similarity = 1 - distance
                benzerlik = max(0.0, 1.0 - float(uzaklik))
                cikti.append((self._belgeler[belge_id], benzerlik))

        return cikti

    def benzer_bul(self, belge_id: str, k: int = 5) -> List[Tuple[Belge, float]]:
        """
        Verilen belgeye benzer diğer belgeleri bulur.
        E-ticarette "bunu alanlar şunu da aldı" özelliği için kullanılır.
        """
        if belge_id not in self._belgeler:
            return []
        belge = self._belgeler[belge_id]
        if not belge.embedding:
            return []
        # Belgenin kendi embedding'ini sorgu vektörü olarak kullan
        sonuclar = self.koleksiyon.query(
            query_embeddings = [belge.embedding],
            n_results        = k + 1,       # Kendisi de dönecek, +1 al
            include          = ["metadatas", "distances"]
        )
        cikti = []
        for bid, uzaklik in zip(
            sonuclar["ids"][0],
            sonuclar["distances"][0]
        ):
            if bid != belge_id and bid in self._belgeler:
                benzerlik = max(0.0, 1.0 - float(uzaklik))
                cikti.append((self._belgeler[bid], benzerlik))
        return cikti[:k]


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5: RERANKER (Çapraz Kodlayıcı)
# ══════════════════════════════════════════════════════════════════════════════

class Reranker:
    """
    İki aşamalı arama mimarisinin ikinci aşaması: Reranking.

    ─── Neden Reranking Gerekir? ────────────────────────────────────────────
    Birinci aşama (Bi-encoder): Hızlı ama kaba
        Sorgu ve belge ayrı ayrı encode edilir → bağımsız vektörler
        10M belgede milisaniyeler içinde çalışır

    İkinci aşama (Cross-encoder): Yavaş ama hassas
        Sorgu + belge BIRLIKTE encode edilir → karşılıklı dikkat
        Çok daha doğru sıralama, ama her çift için ayrı forward pass

    ─── Pratikte ───────────────────────────────────────────────────────────
    1. Bi-encoder ile 1000 adaydan 50 seç (hızlı, ucuz)
    2. Cross-encoder ile 50 adayı doğru sırala (yavaş, pahalı)
    3. Kullanıcıya ilk 10'u göster

    ─── Üretim Reranker Araçları ────────────────────────────────────────────
    • Cohere Rerank API (en kolay): cohere.rerank(...)
    • sentence-transformers: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    • BM25 tabanlı: basit ama etkili
    • LLM tabanlı (en pahalı): dış LLM ile sıralama

    Bu sınıf: Özellik tabanlı sezgisel reranker (API gerektirmez)
    """

    def __init__(self, lambda_istemcisi: YerelAIIstemcisi):
        self.lambda_api = lambda_istemcisi

    def rerank(self, sorgu: str,
               sonuclar: List[AramaSonucu],
               k: int = 10) -> List[AramaSonucu]:
        """
        Çoklu özelliğe dayalı reranking.

        Özellikler ve ağırlıklar:
        • Semantik benzerlik  (%40) — temel vektör benzerliği
        • BM25 tam metin skoru (%25) — kelime eşleşmesi
        • Belge kalite skoru  (%20) — puan + görüntüleme
        • Sorgu kapsamı       (%15) — sorgu kelimelerinin belgede geçişi

        Üretimdeki gerçek cross-encoder:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            skorlar = model.predict([(sorgu, b.icerik) for b in belgeler])
        """
        sorgu_lower = sorgu.lower()
        sorgu_kelimeleri = set(sorgu_lower.split())

        for sonuc in sonuclar:
            b = sonuc.belge

            # ── Özellik 1: Semantik Skor (zaten mevcut) ──────────────────
            sem = sonuc.semantik_skor

            # ── Özellik 2: BM25 Skoru (normalize et) ─────────────────────
            # BM25 skor aralığı değişken, 0-1 arasına sıkıştır
            bm25_norm = min(1.0, sonuc.bm25_skor / 10.0) if sonuc.bm25_skor > 0 else 0.0

            # ── Özellik 3: Belge Kalite Skoru ─────────────────────────────
            # Kullanıcı puanı (0-5) → 0-1 arası
            puan_norm = b.puan / 5.0
            # Görüntüleme sayısı → logaritmik ölçek
            gorum_norm = min(1.0, math.log1p(b.goruntuleme) / 10.0)
            kalite = 0.6 * puan_norm + 0.4 * gorum_norm

            # ── Özellik 4: Sorgu Kapsam Skoru ─────────────────────────────
            # Sorgu kelimelerinin kaçı belgede geçiyor?
            belge_lower = b.tam_metin.lower()
            eslesen = sum(1 for k in sorgu_kelimeleri if k in belge_lower)
            kapsam = eslesen / max(len(sorgu_kelimeleri), 1)

            # ── Bonus: Başlıkta Geçiş ─────────────────────────────────────
            baslik_bonusu = 0.1 if any(k in b.baslik.lower()
                                       for k in sorgu_kelimeleri) else 0.0

            # ── Nihai Skor Hesaplama ───────────────────────────────────────
            sonuc.rerank_skor = (
                0.40 * sem       +
                0.25 * bm25_norm +
                0.20 * kalite    +
                0.15 * kapsam    +
                baslik_bonusu
            )
            # Açıklama ekle (debugging / şeffaflık için)
            sonuc.aciklama = (
                f"sem={sem:.2f} bm25={bm25_norm:.2f} "
                f"kalite={kalite:.2f} kapsam={kapsam:.2f}"
            )

        # Rerank skoruna göre sırala
        sonuclar.sort(key=lambda s: s.rerank_skor, reverse=True)
        return sonuclar[:k]

    def llm_rerank(self, sorgu: str,
                   adaylar: List[AramaSonucu],
                   k: int = 5) -> List[AramaSonucu]:
        """
        Dış LLM ile reranking — en yüksek doğruluk, en yüksek maliyet.

        LLM'den her belge için 0-10 arasında alaka skoru ister.
        Yalnızca k ≤ 20 aday için pratiktir.
        """
        if len(adaylar) == 0:
            return adaylar

        # LLM'e gönderilecek prompt
        belge_listesi = "\n".join([
            f"{i+1}. [{a.belge.baslik}]: {a.belge.icerik[:200]}"
            for i, a in enumerate(adaylar[:20])
        ])
        sistem = (
            "Sen bir arama motoru alaka değerlendirme uzmanısın. "
            "Her belge için 0-10 arasında alaka skoru ver. "
            "Sadece JSON formatında cevap ver: "
            '{"skorlar": [7, 3, 9, ...]}'
        )
        kullanici = (
            f"Sorgu: '{sorgu}'\n\nBelgeler:\n{belge_listesi}\n\n"
            "Her belgeye 0-10 alaka skoru ver."
        )

        # LLM çağrısı (simülasyon veya dış servis)
        cevap = self.lambda_api.llm_cevap_uret(sistem, kullanici)

        try:
            # JSON parse
            import re as re_mod
            json_eslem = re_mod.search(r'\{.*\}', cevap, re_mod.DOTALL)
            if json_eslem:
                veri = json.loads(json_eslem.group())
                llm_skorlar = veri.get("skorlar", [])
                for i, aday in enumerate(adaylar[:len(llm_skorlar)]):
                    aday.rerank_skor = llm_skorlar[i] / 10.0
        except (json.JSONDecodeError, KeyError, IndexError):
            # LLM parse edilemezse mevcut sıralamayla devam et
            log.warning("LLM rerank yanıtı parse edilemedi, atlanıyor")

        adaylar.sort(key=lambda s: s.rerank_skor, reverse=True)
        return adaylar[:k]


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6: HİBRİT ARAMA MOTORU (Ana Orkestratör)
# ══════════════════════════════════════════════════════════════════════════════

class HibritAramaMotoru:
    """
    Semantik + BM25 + Reranking — Üç katmanlı hibrit arama.

    ─── Neden Hibrit? ───────────────────────────────────────────────────────
    Semantik Arama tek başına:
    ✓ "otomobil servisi" → "araba tamiri" bulur
    ✗ "iPhone 15 Pro Max batarya değişim fiyatı" → belirsizleşir

    BM25 tek başına:
    ✓ Ürün kodu, model numarası, isim araması mükemmel
    ✗ Anlam aramasında yetersiz

    Hibrit (ikisi birden):
    ✓ Her ikisinin güçlü yanlarını birleştirir
    ✓ Ticari arama motorları (Google, Elastic) bu yaklaşımı kullanır

    ─── Reciprocal Rank Fusion (RRF) ────────────────────────────────────────
    İki farklı sıralama listesini birleştirmenin en popüler yöntemi.

    RRF(d) = Σ 1 / (k + rank(d))
    k = 60 (sabit, deneysel olarak optimal)

    Örnek:
        Belge A: Semantik sıra=1, BM25 sıra=3
        Belge B: Semantik sıra=2, BM25 sıra=1
        Belge C: Semantik sıra=5, BM25 sıra=2

        RRF(A) = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0156 = 0.0320
        RRF(B) = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
        RRF(C) = 1/(60+5) + 1/(60+2) = 0.0154 + 0.0161 = 0.0315
        → Sıralama: B > A > C
    """

    def __init__(self,
                 lambda_istemcisi : YerelAIIstemcisi,
                 rrf_k            : int = 60,
                 koleksiyon_adi   : str = "semantic_search_demo"):
        self.lambda_api    = lambda_istemcisi
        self.rrf_k         = rrf_k
        self.semantik_motor = SemanticAramaMotoru(
            lambda_istemcisi,
            koleksiyon_adi=koleksiyon_adi
        )
        self.bm25_motor     = BM25Motor()
        self.reranker       = Reranker(lambda_istemcisi)
        self.istatistikler  : List[AramaIstatistigi] = []
        log.info("Hibrit Arama Motoru hazır (Semantik + BM25 + Rerank)")

    def indeksle(self, belgeler: List[Belge]) -> None:
        """Hem semantik hem de BM25 indekslerini oluşturur."""
        log.info(f"Çift indeks oluşturuluyor: {len(belgeler)} belge...")
        self.semantik_motor.indeksle(belgeler)
        self.bm25_motor.indeksle(belgeler)
        log.info("İndeksleme tamamlandı")

    def ara(self,
            sorgu           : str,
            strateji        : str  = "hibrit",
            k               : int  = 10,
            rerank          : bool = True,
            filtreler       : Optional[Dict] = None,
            min_skor        : float = 0.0,
            ) -> Tuple[List[AramaSonucu], AramaIstatistigi]:
        """
        Ana arama metodu.

        sorgu     : Doğal dil araması
        strateji  : "semantik" | "bm25" | "hibrit"
        k         : Kaç sonuç dönsün
        rerank    : Reranking uygulansın mı?
        filtreler : Metadata filtreleri (kategori, tarih, vb.)
        min_skor  : Bu skorun altındaki sonuçları at

        Returns: (sonuçlar, istatistik)
        """
        t_toplam = time.perf_counter()
        aday_havuzu: Dict[str, AramaSonucu] = {}

        # ─── Adım 1: Semantik Arama ────────────────────────────────────────
        t0 = time.perf_counter()
        aday_k = k * 5  # Reranking için geniş aday havuzu
        if strateji in ("semantik", "hibrit"):
            sem_sonuclar = self.semantik_motor.ara(sorgu, k=aday_k,
                                                   filtreler=filtreler)
            for siralama, (belge, skor) in enumerate(sem_sonuclar):
                if belge.id not in aday_havuzu:
                    aday_havuzu[belge.id] = AramaSonucu(
                        belge=belge, eslesen_alan="semantik"
                    )
                aday_havuzu[belge.id].semantik_skor = skor
                # RRF katkısı
                aday_havuzu[belge.id].nihai_skor += 1.0 / (self.rrf_k + siralama + 1)
        sem_ms = (time.perf_counter() - t0) * 1000

        # ─── Adım 2: BM25 Tam Metin Arama ─────────────────────────────────
        t0 = time.perf_counter()
        if strateji in ("bm25", "hibrit"):
            bm25_sonuclar = self.bm25_motor.ara(sorgu, k=aday_k)
            # BM25 skorlarını normalize et (max normalize)
            max_bm25 = max((s for _, s in bm25_sonuclar), default=1.0)
            for siralama, (belge, skor) in enumerate(bm25_sonuclar):
                norm_skor = skor / max_bm25 if max_bm25 > 0 else 0.0
                if belge.id not in aday_havuzu:
                    aday_havuzu[belge.id] = AramaSonucu(
                        belge=belge, eslesen_alan="bm25"
                    )
                else:
                    aday_havuzu[belge.id].eslesen_alan = "hibrit"
                aday_havuzu[belge.id].bm25_skor = norm_skor
                # RRF katkısı ekle
                aday_havuzu[belge.id].nihai_skor += 1.0 / (self.rrf_k + siralama + 1)
        bm25_ms = (time.perf_counter() - t0) * 1000

        # ─── Adım 3: RRF Sıralaması ────────────────────────────────────────
        adaylar = list(aday_havuzu.values())
        adaylar.sort(key=lambda s: s.nihai_skor, reverse=True)

        # ─── Adım 4: Minimum Skor Filtresi ────────────────────────────────
        if min_skor > 0:
            adaylar = [a for a in adaylar
                       if a.semantik_skor >= min_skor or a.bm25_skor >= min_skor]

        # ─── Adım 5: Reranking ────────────────────────────────────────────
        t0 = time.perf_counter()
        if rerank and adaylar:
            adaylar = self.reranker.rerank(sorgu, adaylar[:k*3], k=k)
            # Nihai skoru rerank skoruyla güncelle
            for a in adaylar:
                a.nihai_skor = a.rerank_skor
        else:
            adaylar = adaylar[:k]
        rerank_ms = (time.perf_counter() - t0) * 1000

        # ─── İstatistik Kaydı ─────────────────────────────────────────────
        toplam_ms = (time.perf_counter() - t_toplam) * 1000
        istat = AramaIstatistigi(
            sorgu        = sorgu,
            bulunan_sonuc = len(adaylar),
            semantik_ms  = sem_ms,
            bm25_ms      = bm25_ms,
            rerank_ms    = rerank_ms,
            toplam_ms    = toplam_ms,
            strateji     = strateji,
        )
        self.istatistikler.append(istat)

        return adaylar, istat

    def benzer_bul(self, belge_id: str, k: int = 5) -> List[AramaSonucu]:
        """'Buna benzer belgeler' özelliği."""
        sonuclar = self.semantik_motor.benzer_bul(belge_id, k)
        return [
            AramaSonucu(belge=b, semantik_skor=s, nihai_skor=s,
                        eslesen_alan="benzerlik")
            for b, s in sonuclar
        ]

    def performans_raporu(self) -> Dict[str, Any]:
        """Son N sorgunun performans özeti."""
        if not self.istatistikler:
            return {}
        toplam = len(self.istatistikler)
        return {
            "toplam_sorgu"   : toplam,
            "ort_ms"         : round(sum(i.toplam_ms for i in self.istatistikler) / toplam, 2),
            "ort_sem_ms"     : round(sum(i.semantik_ms for i in self.istatistikler) / toplam, 2),
            "ort_bm25_ms"    : round(sum(i.bm25_ms for i in self.istatistikler) / toplam, 2),
            "ort_rerank_ms"  : round(sum(i.rerank_ms for i in self.istatistikler) / toplam, 2),
            "ort_sonuc"      : round(sum(i.bulunan_sonuc for i in self.istatistikler) / toplam, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7: RAG SİSTEMİ (Retrieval-Augmented Generation)
# ══════════════════════════════════════════════════════════════════════════════

class RAGSistemi:
    """
    Yerel LLM simülasyonu + Semantik Arama ile soru-cevap sistemi.

    RAG Neden Gereklidir?
    ─────────────────────
    LLM'ler eğitim kesim tarihinden sonrasını bilmez.
    Şirket/organizasyon spesifik bilgiye sahip değildir.
    Halüsinasyon riski taşırlar.

    RAG bu sorunları çözer:
    1. Soruyla ilgili gerçek belgeleri getir
    2. LLM'e "bu belgelerden yararlanarak cevapla" de
    3. LLM artık belgeye dayalı, doğrulanabilir cevap verir

    Örnek: "Ürünümün garantisi ne zaman biter?"
    → Garanti belgeleri vektör DB'de
    → Kullanıcıya özel kayıt bulunur
    → LLM garanti belgesine bakarak kesin cevap verir
    """

    def __init__(self, arama_motoru: HibritAramaMotoru,
                 lambda_istemcisi: YerelAIIstemcisi):
        self.arama  = arama_motoru
        self.lambda_api = lambda_istemcisi
        self.geçmiş: List[Dict[str, str]] = []   # Çok turlu sohbet

    def cevapla(self, soru: str,
                bagiam_k: int = 3,
                min_benzerlik: float = 0.3,
                sohbet_modu: bool = False) -> Dict[str, Any]:
        """
        Soruyu hibrit arama ile cevaplar.

        soru           : Kullanıcı sorusu
        bagiam_k       : Kaç belge bağlam olarak kullanılsın
        min_benzerlik  : Bu değerin altındaki belgeler bağlam dışı bırakılır
        sohbet_modu    : Önceki turları bağlam olarak ekle
        """
        t_baslangic = time.perf_counter()

        # ─── Adım 1: İlgili Belgeleri Getir ───────────────────────────────
        sonuclar, istat = self.arama.ara(
            sorgu    = soru,
            strateji = "hibrit",
            k        = bagiam_k,
            rerank   = True,
            min_skor = min_benzerlik,
        )

        # ─── Adım 2: Bağlam Oluştur ───────────────────────────────────────
        if sonuclar:
            bagiam_parcalari = []
            for i, sonuc in enumerate(sonuclar):
                b = sonuc.belge
                bagiam_parcalari.append(
                    f"[Kaynak {i+1}: {b.baslik} | {b.kategori} | {b.tarih}]\n"
                    f"{b.icerik}"
                )
            bagiam_metni = "\n\n---\n\n".join(bagiam_parcalari)
        else:
            bagiam_metni = ""

        # ─── Adım 3: LLM Prompt Hazırla ───────────────────────────────────
        sistem_mesaji = """Sen yardımcı bir asistansın. Sana verilen belgelerden
yararlanarak soruları cevapla. Belgede bilgi yoksa bunu belirt.
Cevabını Türkçe ver ve kullandığın kaynakları belirt."""

        if bagiam_metni:
            kullanici_mesaji = (
                f"Aşağıdaki belgeler bulundu:\n\n{bagiam_metni}\n\n"
                f"Soru: {soru}"
            )
        else:
            kullanici_mesaji = (
                f"İlgili belge bulunamadı.\nSoru: {soru}"
            )

        # ─── Adım 4: Yerel LLM Simülasyon Çağrısı ─────────────────────────
        llm_cevabi = self.lambda_api.llm_cevap_uret(
            sistem   = sistem_mesaji,
            kullanici = kullanici_mesaji,
            sicaklik  = 0.3,   # Düşük sıcaklık = tutarlı/belirleyici cevap
        )

        # ─── Adım 5: Sohbet Geçmişi ───────────────────────────────────────
        if sohbet_modu:
            self.geçmiş.append({"role": "user",      "content": soru})
            self.geçmiş.append({"role": "assistant", "content": llm_cevabi})

        toplam_ms = (time.perf_counter() - t_baslangic) * 1000

        return {
            "soru"         : soru,
            "cevap"        : llm_cevabi,
            "kaynaklar"    : [s.belge.baslik for s in sonuclar],
            "kaynak_sayisi": len(sonuclar),
            "sure_ms"      : round(toplam_ms, 2),
            "istatistik"   : {
                "semantik_ms": round(istat.semantik_ms, 2),
                "bm25_ms"    : round(istat.bm25_ms, 2),
                "rerank_ms"  : round(istat.rerank_ms, 2),
            }
        }


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 8: TEST VERİ SETİ
# ══════════════════════════════════════════════════════════════════════════════

def test_veri_seti_olustur() -> List[Belge]:
    """
    Gerçekçi test belgelerinden oluşan veri seti.
    Üretimde bu veriler PostgreSQL, MongoDB veya S3'ten gelir.
    """
    belgeler_ham = [
        # ─── Teknoloji ────────────────────────────────────────────────────
        ("B001", "Python ile Veri Bilimi",
         "Python programlama dili veri bilimi ve makine öğrenmesi için "
         "en popüler dildir. Pandas, NumPy, Scikit-learn kütüphaneleri "
         "ile veri analizi, görselleştirme ve model eğitimi kolayca yapılabilir. "
         "Jupyter Notebook ile interaktif geliştirme ortamı sağlanır.",
         "teknoloji", "Dr. Ali Kaya", "2024-03-15",
         ["python", "veri-bilimi", "pandas", "makine-öğrenmesi"], 4.8, 15420),

        ("B002", "Yapay Zeka ve Derin Öğrenme",
         "Derin öğrenme yapay sinir ağları kullanarak büyük veri setlerinden "
         "otomatik özellik öğrenme yöntemidir. TensorFlow ve PyTorch framework'leri "
         "ile konvolüsyonel ağlar, transformerlar ve LSTM modelleri oluşturulabilir. "
         "GPT, BERT ve LLaMA modelleri bu teknolojiyle geliştirilmiştir.",
         "teknoloji", "Dr. Elif Şahin", "2024-02-20",
         ["yapay-zeka", "derin-öğrenme", "tensorflow", "pytorch"], 4.9, 22100),

        ("B003", "Kubernetes ile Mikroservis Mimarisi",
         "Kubernetes konteyner orkestrasyon platformu ile mikroservis mimarisi "
         "ölçeklenebilir şekilde yönetilebilir. Pod, Deployment, Service ve Ingress "
         "kaynakları kullanılarak yüksek erişilebilirlik sağlanır. "
         "Helm chart'ları ile uygulama dağıtımı standartlaştırılır.",
         "teknoloji", "Murat Arslan", "2024-01-10",
         ["kubernetes", "docker", "mikroservis", "devops"], 4.6, 8900),

        ("B004", "Vektör Veritabanları ve Semantik Arama",
         "Vektör veritabanları (ChromaDB, Milvus, Pinecone) embedding vektörlerini "
         "depolayarak anlam tabanlı arama sağlar. HNSW ve IVF algoritmaları "
         "milyarlarca vektörde milisaniye cevap süreleriyle yaklaşık k-NN araması yapar. "
         "RAG sistemlerinin temel altyapısını oluşturur.",
         "teknoloji", "Zeynep Çelik", "2024-04-01",
         ["vektör-db", "embedding", "semantik-arama", "rag"], 4.7, 11200),

        ("B005", "FastAPI ile Yüksek Performanslı REST API",
         "FastAPI Python tabanlı modern web framework'üdür. Pydantic ile veri "
         "doğrulama, async/await ile asenkron işleme ve OpenAPI ile otomatik "
         "API dokümantasyonu sağlar. Starlette üzerine kurulmuş olup "
         "saniyede binlerce isteği işleyebilir.",
         "teknoloji", "Hasan Çelik", "2024-03-22",
         ["fastapi", "python", "rest-api", "async"], 4.5, 9800),

        ("B006", "GPU Bulut Altyapısı ve Model Servisleri",
         "Yüksek performanslı GPU bulut hizmetleri ve OpenAI uyumlu "
         "LLM API sunar. A100, H100 GPU'larla model eğitimi ve inference desteklenir. "
         "LLaMA, Falcon, Mistral gibi açık kaynak modeller kullanılabilir. "
         "Veri gizliliği için on-premise alternatifi de mevcuttur.",
         "teknoloji", "Lambda Ekibi", "2024-04-15",
         ["lambda-labs", "gpu", "llm", "cloud", "api"], 4.7, 7600),

        # ─── Sağlık ───────────────────────────────────────────────────────
        ("B007", "Tip 2 Diyabet Tedavisi",
         "Tip 2 diyabet insülin direnci ile karakterize kronik metabolik hastalıktır. "
         "Tedavide yaşam tarzı değişiklikleri, beslenme düzenlemesi ve metformin "
         "başta olmak üzere oral antidiyabetik ilaçlar kullanılır. "
         "Düzenli HbA1c takibi ve komplikasyon taraması hayati önem taşır.",
         "sağlık", "Prof. Dr. Fatma Yıldız", "2024-02-28",
         ["diyabet", "insülin", "tedavi", "metabolizma"], 4.9, 31400),

        ("B008", "Kalp Sağlığı ve Kardiyovasküler Hastalıklar",
         "Kardiyovasküler hastalıklar dünya genelinde önde gelen ölüm nedenidir. "
         "Risk faktörleri: hipertansiyon, hiperlipidemi, sigara, hareketsiz yaşam. "
         "Akdeniz diyeti, düzenli aerobik egzersiz ve stres yönetimi "
         "koruyucu faktörlerin başında gelir.",
         "sağlık", "Prof. Dr. Ahmet Öz", "2024-01-30",
         ["kalp", "kardiyovasküler", "hipertansiyon", "beslenme"], 4.8, 28700),

        ("B009", "Bağışıklık Sistemi ve Bağırsak Mikrobiyomu",
         "Bağırsak mikrobiyomun %70'i bağışıklık sistemini etkiler. "
         "Probiyotikler ve prebiyotikler bağırsak florasını destekler. "
         "Fermente gıdalar (yoğurt, kefir, turşu) ve lif açısından zengin diyet "
         "mikrobiyom çeşitliliğini artırır.",
         "sağlık", "Dr. Selin Kara", "2024-03-05",
         ["bağışıklık", "mikrobiyom", "probiyotik", "beslenme"], 4.6, 18900),

        # ─── Finans ───────────────────────────────────────────────────────
        ("B010", "Borsa Yatırımı için Temel Analiz",
         "Temel analizde şirketin finansal tabloları, kazanç raporları ve "
         "sektör dinamikleri incelenir. F/K oranı, F/DD oranı ve EBITDA "
         "kritik göstergelerdir. Warren Buffett yaklaşımı olan değer yatırımı "
         "uzun vadeli büyüme potansiyelini esas alır.",
         "finans", "Cengiz Bora", "2024-03-18",
         ["borsa", "hisse", "temel-analiz", "yatırım"], 4.4, 14200),

        ("B011", "Kripto Para Teknolojisi ve Blockchain",
         "Blockchain merkezi olmayan dağıtık defter teknolojisidir. "
         "Bitcoin ve Ethereum proof-of-work ve proof-of-stake konsensüs mekanizmaları kullanır. "
         "DeFi, NFT ve akıllı sözleşmeler Web3 ekosisteminin temelini oluşturur. "
         "Kripto piyasası yüksek volatilite içerir.",
         "finans", "Serhan Yılmaz", "2024-04-10",
         ["bitcoin", "ethereum", "blockchain", "kripto", "defi"], 4.3, 19800),

        ("B012", "Kişisel Finans Yönetimi",
         "Kişisel finans yönetiminin temeli bütçe oluşturmaktır. "
         "50/30/20 kuralı: gelirin %50 ihtiyaçlara, %30 isteklere, %20 tasarrufa. "
         "Acil fon olarak 3-6 aylık gider tutarı ayrılmalı, "
         "borçlanmada düşük faiz önceliği gözetilmelidir.",
         "finans", "Pınar Akar", "2024-02-14",
         ["tasarruf", "bütçe", "kişisel-finans", "yatırım"], 4.6, 23100),

        # ─── Hukuk ────────────────────────────────────────────────────────
        ("B013", "İş Hukuku: Çalışan Hakları",
         "İş Kanunu'na göre çalışanlar fazla mesai ücreti, yıllık izin ve kıdem "
         "tazminatı haklarına sahiptir. İş akdi feshedilirken yasal prosedürlere "
         "uyulmalıdır. Mobing, haksız fesih ve iş kazaları iş mahkemelerinde "
         "dava konusu olabilir.",
         "hukuk", "Av. Deniz Güven", "2024-01-25",
         ["iş-hukuku", "tazminat", "çalışan-hakları", "iş-kanunu"], 4.7, 12300),

        # ─── Bilim ────────────────────────────────────────────────────────
        ("B014", "Kuantum Bilişim Temelleri",
         "Kuantum bilgisayarlar qubit kullanarak klasik bilgisayarların "
         "üstesinden gelemeyeceği problemleri çözer. Süperpozisyon ve dolanıklık "
         "ilkeleri sayesinde paralel hesaplama yapılabilir. "
         "IBM, Google ve Microsoft bu alanda rekabet etmektedir.",
         "bilim", "Prof. Dr. Kaan Özlü", "2024-03-28",
         ["kuantum", "qubit", "hesaplama", "fizik"], 4.8, 9400),

        ("B015", "İklim Değişikliği ve Yenilenebilir Enerji",
         "Küresel ısınmanın 1.5°C ile sınırlandırılması için 2050'ye kadar "
         "net sıfır emisyon hedeflenmektedir. Güneş, rüzgar ve hidroelektrik "
         "yenilenebilir enerji kaynaklarının payı artmaktadır. "
         "Karbon vergilendirmesi politika aracı olarak yaygınlaşmaktadır.",
         "bilim", "Dr. Burcu Demir", "2024-04-05",
         ["iklim", "enerji", "sürdürülebilirlik", "karbon"], 4.7, 17600),
    ]

    belgeler = []
    for (bid, baslik, icerik, kategori, yazar, tarih, etiketler, puan, gorum) in belgeler_ham:
        belgeler.append(Belge(
            id=bid, baslik=baslik, icerik=icerik,
            kategori=kategori, yazar=yazar, tarih=tarih,
            etiketler=etiketler, puan=puan, goruntuleme=gorum
        ))
    return belgeler


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 9: UNIT TESTLER
# ══════════════════════════════════════════════════════════════════════════════

class SemanticAramaTestleri(unittest.TestCase):
    """
    Arama sisteminin doğruluğunu otomatik test eder.

    Üretimde bu testler CI/CD pipeline'ına entegre edilir:
        pytest tests/test_search.py -v --cov=semantic_search
    """

    @classmethod
    def setUpClass(cls):
        """Tüm testler için tek seferlik kurulum."""
        import logging
        logging.disable(logging.CRITICAL)
        cls.lambda_api = YerelAIIstemcisi(simule=True)
        test_koleksiyon_adi = f"semantic_search_test_{int(time.time() * 1000)}"
        cls.motor = HibritAramaMotoru(
            cls.lambda_api,
            koleksiyon_adi=test_koleksiyon_adi
        )
        cls.motor.indeksle(test_veri_seti_olustur())
        logging.disable(logging.NOTSET)

    def test_embedding_boyutu(self):
        """Embedding vektörünün doğru boyutta üretildiğini kontrol eder."""
        vec = self.lambda_api.embedding_uret(["test metni"])[0]
        self.assertEqual(len(vec), YerelAIIstemcisi.EMBED_DIM)

    def test_embedding_normalize(self):
        """Embedding vektörünün birim vektör olduğunu kontrol eder."""
        vec = np.array(self.lambda_api.embedding_uret(["normalize test"])[0])
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_semantik_benzerlik_dogru_yon(self):
        """Benzer metinlerin farklı metinlere göre daha yüksek skor aldığını test eder."""
        api = self.lambda_api
        ref     = api.embedding_uret(["Python makine öğrenmesi"])[0]
        yakin   = api.embedding_uret(["yapay zeka derin öğrenme"])[0]
        uzak    = api.embedding_uret(["futbol maçı"])[0]

        ref_a, yakin_a, uzak_a = np.array(ref), np.array(yakin), np.array(uzak)
        skor_yakin = float(np.dot(ref_a, yakin_a))
        skor_uzak  = float(np.dot(ref_a, uzak_a))
        self.assertGreater(skor_yakin, skor_uzak,
            "Yakın metinlerin benzerliği uzak metinlerin benzerliğinden büyük olmalı")

    def test_semantik_arama_sonuc_donuyor(self):
        """Semantik aramanın sonuç döndürdüğünü kontrol eder."""
        sonuclar, _ = self.motor.ara("python programlama", k=3)
        self.assertGreater(len(sonuclar), 0)
        self.assertLessEqual(len(sonuclar), 3)

    def test_bm25_tam_kelime_buluyor(self):
        """BM25'in tam kelime eşleştirme yaptığını test eder."""
        sonuclar = self.motor.bm25_motor.ara("Python veri bilimi", k=5)
        self.assertGreater(len(sonuclar), 0)
        # En üst sonucun içeriğinde python veya veri geçmeli
        ust_belge = sonuclar[0][0]
        self.assertTrue(
            "python" in ust_belge.tam_metin.lower() or
            "veri" in ust_belge.tam_metin.lower()
        )

    def test_hibrit_sonuc_sayisi(self):
        """Hibrit aramanın istenen sayıda sonuç döndürdüğünü test eder."""
        for k in [1, 5, 10]:
            sonuclar, _ = self.motor.ara("teknoloji", k=k)
            self.assertLessEqual(len(sonuclar), k)

    def test_metadata_filtreleme(self):
        """Metadata filtrelemesinin doğru çalıştığını test eder."""
        sonuclar, _ = self.motor.ara(
            "makine öğrenmesi",
            filtreler={"kategori": {"$eq": "teknoloji"}},
            k=5
        )
        for s in sonuclar:
            self.assertEqual(s.belge.kategori, "teknoloji")

    def test_bos_sorgu_guvenli(self):
        """Boş veya tek kelimeli sorguların çökmediğini test eder."""
        for sorgu in ["", " ", "a"]:
            try:
                sonuclar, _ = self.motor.ara(sorgu, k=3)
                # Sonuç sayısı fark etmez, çökmemeli
            except Exception as e:
                self.fail(f"Boş sorgu exception fırlattı: {e}")

    def test_benzer_belge_kendini_icermiyor(self):
        """'Benzer belgeler' aramasının aynı belgeyi döndürmediğini test eder."""
        sonuclar = self.motor.benzer_bul("B001", k=5)
        ids = [s.belge.id for s in sonuclar]
        self.assertNotIn("B001", ids)


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 10: DEMO — TÜM ÖZELLİKLER
# ══════════════════════════════════════════════════════════════════════════════

def demo_calistir():
    """Tüm sistem özelliklerini gösteren kapsamlı demo."""
    print("\n" + "█" * 68)
    print("  SEMANTİK ARAMA — Yerel Simülasyon ile Üretim Sistemi Demo")
    print("█" * 68)

    # ─── Sistem Kurulumu ──────────────────────────────────────────────────
    print("\n⚙️  Sistem Kuruluyor...")
    lambda_api = YerelAIIstemcisi(simule=True)
    motor = HibritAramaMotoru(lambda_api)
    belgeler = test_veri_seti_olustur()
    motor.indeksle(belgeler)
    rag = RAGSistemi(motor, lambda_api)
    print(f"  ✓ {len(belgeler)} belge indekslendi")
    print(f"  ✓ Yerel simülasyon motoru: {'Aktif'}")
    print(f"  ✓ Semantik + BM25 + Reranker hazır\n")

    # ─────────────────────────────────────────────────────────────────────
    # DEMO 1: Strateji Karşılaştırması
    # ─────────────────────────────────────────────────────────────────────
    print("═" * 68)
    print("  DEMO 1: Arama Stratejisi Karşılaştırması")
    print("═" * 68)

    sorgu = "yapay zeka ve makine öğrenmesi algoritmaları"
    print(f"\n  📝 Sorgu: '{sorgu}'\n")

    for strateji, rerank in [("semantik", False), ("bm25", False), ("hibrit", True)]:
        sonuclar, istat = motor.ara(sorgu, strateji=strateji,
                                    k=3, rerank=rerank)
        print(f"  ── {strateji.upper()} {'+ Rerank' if rerank else '':8} "
              f"({istat.toplam_ms:.1f}ms) ──")
        for i, s in enumerate(sonuclar):
            etiket = {
                "semantik": f"sem={s.semantik_skor:.3f}",
                "bm25"    : f"bm25={s.bm25_skor:.3f}",
                "hibrit"  : f"nihai={s.nihai_skor:.3f}",
            }[strateji]
            print(f"  {i+1}. [{etiket}] {s.belge.baslik}")
        print()

    # ─────────────────────────────────────────────────────────────────────
    # DEMO 2: Semantik Anlama — Eş Anlamlı Sorgular
    # ─────────────────────────────────────────────────────────────────────
    print("═" * 68)
    print("  DEMO 2: Semantik Anlama (Eş Anlamlılar)")
    print("═" * 68)
    print("\n  Farklı ifadeler, aynı anlam → Aynı belgeler bulunmalı\n")

    es_anlamli_sorgular = [
        "python kod yazımı",
        "bilgisayar programlama dili",
        "yazılım geliştirme tekniği",
    ]
    for sorgu in es_anlamli_sorgular:
        sonuclar, istat = motor.ara(sorgu, strateji="semantik", k=1)
        if sonuclar:
            s = sonuclar[0]
            print(f"  🔍 '{sorgu}'")
            print(f"     → {s.belge.baslik} "
                  f"(benzerlik: {s.semantik_skor:.3f}, {istat.toplam_ms:.1f}ms)\n")

    # ─────────────────────────────────────────────────────────────────────
    # DEMO 3: Metadata Filtrelemeli Hibrit Arama
    # ─────────────────────────────────────────────────────────────────────
    print("═" * 68)
    print("  DEMO 3: Metadata Filtrelemeli Arama")
    print("═" * 68)

    filtre_testleri = [
        ("blockchain kripto", {"kategori": {"$eq": "finans"}},
         "Finans kategorisinde blockchain"),
        ("tedavi ve sağlık", {"kategori": {"$eq": "sağlık"}},
         "Sağlık kategorisinde tedavi"),
        ("veri analizi", {"puan": {"$gte": 4.7}},
         "Puan ≥ 4.7 olan belgeler"),
    ]

    for sorgu, filtre, aciklama in filtre_testleri:
        sonuclar, istat = motor.ara(sorgu, filtreler=filtre,
                                    k=3, rerank=True)
        print(f"\n  📌 {aciklama}")
        print(f"  Sorgu: '{sorgu}' | {istat.toplam_ms:.1f}ms")
        for s in sonuclar:
            print(f"  → [{s.belge.kategori}] {s.belge.baslik} "
                  f"(puan:{s.belge.puan}, skor:{s.nihai_skor:.3f})")

    # ─────────────────────────────────────────────────────────────────────
    # DEMO 4: Benzer Belge Bulma
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  DEMO 4: 'Buna Benzer Belgeler' Özelliği")
    print("═" * 68)

    referans_id = "B002"
    referans_belge = next(b for b in belgeler if b.id == referans_id)
    print(f"\n  Referans: [{referans_id}] {referans_belge.baslik}")
    print(f"  {'─'*50}")
    benzerler = motor.benzer_bul(referans_id, k=4)
    for s in benzerler:
        bar = "█" * int(s.semantik_skor * 15)
        print(f"  [{s.belge.id}] {s.belge.baslik:<40}"
              f"  {s.semantik_skor:.3f} {bar}")

    # ─────────────────────────────────────────────────────────────────────
    # DEMO 5: RAG Sistemi
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  DEMO 5: RAG Soru-Cevap Sistemi")
    print("═" * 68)

    rag_sorulari = [
        "Yerelde embedding üretimi nasıl yapılır?",
        "Diyabet tedavisinde neler kullanılır?",
        "Kuantum bilgisayarlar ne işe yarar?",
    ]

    for soru in rag_sorulari:
        print(f"\n  👤 Soru: '{soru}'")
        yanit = rag.cevapla(soru, bagiam_k=2)
        print(f"  🤖 Yanıt: {yanit['cevap']}")
        print(f"  📚 Kaynaklar ({yanit['kaynak_sayisi']}): "
              f"{' | '.join(yanit['kaynaklar'])}")
        print(f"  ⏱  Süre: {yanit['sure_ms']:.1f}ms")

    # ─────────────────────────────────────────────────────────────────────
    # DEMO 6: Performans Analizi
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  DEMO 6: Performans ve İstatistikler")
    print("═" * 68)

    # Ek sorgularla raporu zenginleştir
    ekstra_sorgular = [
        "blockchain ve kripto para",
        "iş hukuku ve çalışan hakları",
        "iklim değişikliği yenilenebilir enerji",
        "fastapi rest api geliştirme",
        "mikrobiyom bağışıklık sistemi",
    ]
    for s in ekstra_sorgular:
        motor.ara(s, k=5)

    rapor = motor.performans_raporu()
    print(f"\n  📊 Performans Özeti ({rapor['toplam_sorgu']} sorgu):")
    print(f"  {'─'*45}")
    print(f"  Ortalama toplam süre   : {rapor['ort_ms']:>8.2f}ms")
    print(f"  Ortalama semantik süre : {rapor['ort_sem_ms']:>8.2f}ms")
    print(f"  Ortalama BM25 süre     : {rapor['ort_bm25_ms']:>8.2f}ms")
    print(f"  Ortalama rerank süre   : {rapor['ort_rerank_ms']:>8.2f}ms")
    print(f"  Ortalama sonuç sayısı  : {rapor['ort_sonuc']:>8.1f}")

    maliyet = lambda_api.maliyet_raporu()
    print(f"\n  💰 API Kullanım Raporu:")
    print(f"  Önbelleklenen embedding: {maliyet['onbellekte']}")
    print(f"  API çağrısı            : {maliyet['api_cagrisi']}")

    # ─────────────────────────────────────────────────────────────────────
    # DEMO 7: Unit Testler
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  DEMO 7: Otomatik Testler")
    print("═" * 68 + "\n")

    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(SemanticAramaTestleri)
    import io
    runner = unittest.TextTestRunner(verbosity=1, stream=io.StringIO())
    # Sonuçları kendimiz raporlayalım
    sonuc  = runner.run(suite)
    toplam = sonuc.testsRun
    hatali = len(sonuc.failures) + len(sonuc.errors)
    basarili = toplam - hatali
    print(f"  Çalıştırılan test  : {toplam}")
    print(f"  ✅ Başarılı        : {basarili}")
    print(f"  ❌ Başarısız       : {hatali}")
    if sonuc.failures:
        for _, msg in sonuc.failures:
            print(f"  HATA: {msg[:100]}")
    if sonuc.errors:
        for _, msg in sonuc.errors:
            print(f"  ERROR: {msg[:100]}")

    # ─── Genel Özet ───────────────────────────────────────────────────────
    print("\n" + "█" * 68)
    print("  ✅ Demo tamamlandı!")
    print("  " + "─" * 64)
    print("  Yerel kullanım için:")
    print("    1. Bu dosyayı doğrudan çalıştırın")
    print("    2. Simülasyon çıktılarıyla arama kalitesini test edin")
    print("    3. İsteğe bağlı: YerelAIIstemcisi(..., uzak_servis_izinli=True)")
    print("█" * 68 + "\n")


# Geriye dönük uyumluluk için eski isim alias olarak bırakıldı.
LambdaLabsIstemcisi = YerelAIIstemcisi


# ══════════════════════════════════════════════════════════════════════════════
# GİRİŞ NOKTASI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_calistir()
