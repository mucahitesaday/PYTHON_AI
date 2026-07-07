"""
============================================================
  RAG (Retrieval-Augmented Generation) Mimarisi
  ============================================================
  RAG, büyük dil modellerinin (LLM) harici bilgi kaynaklarından
  bilgi alarak yanıt üretmesini sağlayan bir yapay zeka tekniğidir.

  Akış:
  1. Belgeler parçalara (chunk) bölünür
  2. Parçalar vektöre (embedding) dönüştürülür
  3. Vektörler bir veritabanına kaydedilir
  4. Kullanıcı sorusu da vektöre dönüştürülür
  5. En benzer parçalar veritabanından getirilir (retrieval)
  6. Parçalar + soru → LLM'e gönderilir → yanıt üretilir

  Gerekli kütüphaneler:
    pip install langchain langchain-community langchain-openai
    pip install faiss-cpu tiktoken pypdf sentence-transformers
============================================================
"""

# ── Standart kütüphaneler ─────────────────────────────────
import os                        # İşletim sistemi dosya/dizin işlemleri
import logging                   # Log mesajları yazdırmak için
from pathlib import Path         # Dosya yollarını nesne olarak yönetmek için
from typing import List, Optional  # Tip ipuçları (type hints)

# ── Belge yükleme ve bölme ───────────────────────────────
from langchain_community.document_loaders import (
    PyPDFLoader,                 # PDF dosyalarını yükler
    TextLoader,                  # Düz metin (.txt) dosyalarını yükler
    DirectoryLoader              # Bir klasördeki tüm dosyaları yükler
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter: metni özyinelemeli şekilde parçalara böler;
# önce paragraf, sonra cümle, sonra kelime bazında böler → anlam kaybı az olur

# ── Embedding (vektör dönüşümü) ──────────────────────────
from langchain_openai import OpenAIEmbeddings
# OpenAIEmbeddings: metni sayısal vektöre çevirir (text-embedding-ada-002 modeli)
# Alternatif: HuggingFaceEmbeddings (ücretsiz, yerel çalışır)

# ── Vektör veritabanı ────────────────────────────────────
from langchain_community.vectorstores import FAISS
# FAISS (Facebook AI Similarity Search): vektörleri bellekte saklar ve
# kosinüs/L2 benzerliğiyle hızlı arama yapar. Üretim için Pinecone/Weaviate tercih edilir.

# ── Dil modeli (LLM) ─────────────────────────────────────
from langchain_openai import ChatOpenAI
# ChatOpenAI: OpenAI'nin GPT modellerine bağlanır (gpt-4o, gpt-3.5-turbo vb.)

# ── RAG zinciri bileşenleri ──────────────────────────────
from langchain.chains import RetrievalQA
# RetrievalQA: soru + alınan belgeler → LLM → yanıt zinciri oluşturur

from langchain.prompts import PromptTemplate
# PromptTemplate: LLM'e gönderilecek istemi (prompt) şablonlaştırır

from langchain.schema import Document
# Document: LangChain'in temel belge nesnesi {page_content, metadata}

# ── Log ayarları ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,          # INFO ve üstü mesajları göster
    format="%(asctime)s [%(levelname)s] %(message)s"  # Zaman damgalı format
)
logger = logging.getLogger(__name__)  # Bu modüle özel logger nesnesi


# ─────────────────────────────────────────────────────────
#  1. BELGE YÜKLEYİCİ
# ─────────────────────────────────────────────────────────
class BelgeYukleyici:
    """
    Farklı formatlardaki belgeleri (PDF, TXT, klasör) yükleyen sınıf.
    Her yükleme yöntemi LangChain Document nesneleri listesi döndürür.
    """

    @staticmethod
    def pdf_yukle(dosya_yolu: str) -> List[Document]:
        """
        Tek bir PDF dosyasını yükler.

        Args:
            dosya_yolu: PDF dosyasının tam yolu (ör: '/belgeler/rapor.pdf')

        Returns:
            Her sayfayı ayrı Document olarak içeren liste
        """
        logger.info(f"PDF yükleniyor: {dosya_yolu}")  # İşlem logu

        yukleyici = PyPDFLoader(dosya_yolu)  # PyPDF ile PDF aç
        belgeler = yukleyici.load()           # Tüm sayfaları yükle
        # Her Document'ın metadata'sında: {'source': ..., 'page': N} bulunur

        logger.info(f"{len(belgeler)} sayfa yüklendi.")
        return belgeler

    @staticmethod
    def txt_yukle(dosya_yolu: str) -> List[Document]:
        """
        Düz metin (.txt) dosyasını yükler.

        Args:
            dosya_yolu: TXT dosyasının tam yolu

        Returns:
            Tek elemanlı Document listesi (tüm metin tek belgede)
        """
        logger.info(f"TXT yükleniyor: {dosya_yolu}")

        yukleyici = TextLoader(dosya_yolu, encoding="utf-8")  # UTF-8 ile oku
        belgeler = yukleyici.load()

        logger.info(f"{len(belgeler)} belge yüklendi.")
        return belgeler

    @staticmethod
    def klasor_yukle(klasor_yolu: str, glob_pattern: str = "**/*.txt") -> List[Document]:
        """
        Bir klasördeki tüm eşleşen dosyaları yükler.

        Args:
            klasor_yolu  : Taranacak klasör yolu (ör: './veri')
            glob_pattern : Dosya filtresi (ör: '**/*.pdf', '**/*.txt')
                           ** → özyinelemeli klasör taraması

        Returns:
            Tüm belgelerden oluşan birleşik liste
        """
        logger.info(f"Klasör yükleniyor: {klasor_yolu} | Pattern: {glob_pattern}")

        yukleyici = DirectoryLoader(
            klasor_yolu,             # Taranacak kök klasör
            glob=glob_pattern,       # Dosya filtre deseni
            loader_cls=TextLoader    # Her dosyayı TextLoader ile aç
        )
        belgeler = yukleyici.load()  # Tüm eşleşen dosyaları yükle

        logger.info(f"Toplam {len(belgeler)} belge yüklendi.")
        return belgeler


# ─────────────────────────────────────────────────────────
#  2. METİN BÖLÜCÜ (CHUNKING)
# ─────────────────────────────────────────────────────────
class MetinBolucu:
    """
    Uzun belgeleri LLM'in context penceresine sığacak
    küçük parçalara (chunk) bölen sınıf.
    """

    def __init__(
        self,
        parca_boyutu: int = 1000,    # Her parçanın maksimum karakter sayısı
        parca_otusmesi: int = 200    # Ardışık parçalar arasındaki örtüşme miktarı
        # Örtüşme: bağlamın korunması için parçaların birbirini bir miktar tekrarlaması
    ):
        self.parca_boyutu = parca_boyutu
        self.parca_otusmesi = parca_otusmesi

        # RecursiveCharacterTextSplitter: sırasıyla \n\n, \n, ' ', '' ile böler
        # → paragraf → satır → kelime → karakter
        self.bolucu = RecursiveCharacterTextSplitter(
            chunk_size=self.parca_boyutu,       # Max karakter/parça
            chunk_overlap=self.parca_otusmesi,  # Parçalar arası örtüşme
            length_function=len,                # Uzunluk ölçüm fonksiyonu
            separators=["\n\n", "\n", " ", ""]  # Bölme öncelik sırası
        )

    def parcala(self, belgeler: List[Document]) -> List[Document]:
        """
        Belge listesini parçalara böler.

        Args:
            belgeler: BelgeYukleyici'den gelen Document listesi

        Returns:
            Parçalanmış Document listesi (her parça bağımsız Document)
        """
        logger.info(f"{len(belgeler)} belge parçalanıyor...")

        parcalar = self.bolucu.split_documents(belgeler)
        # split_documents: Document'ları böler ve metadata'yı korur

        logger.info(
            f"{len(parcalar)} parça oluşturuldu. "
            f"(Ort. boyut: ~{self.parca_boyutu} karakter)"
        )
        return parcalar


# ─────────────────────────────────────────────────────────
#  3. VEKTÖR VERİTABANI (EMBEDDING + DEPOLAMA)
# ─────────────────────────────────────────────────────────
class VektorDepo:
    """
    Metin parçalarını vektöre çevirip FAISS'e kaydeden,
    daha sonra yükleyen ve benzerlik araması yapan sınıf.
    """

    def __init__(self, openai_api_key: str):
        """
        Args:
            openai_api_key: OpenAI API anahtarı (embedding için gerekli)
        """
        # OpenAI text-embedding-ada-002 modeli: 1536 boyutlu vektör üretir
        self.embedding_modeli = OpenAIEmbeddings(
            openai_api_key=openai_api_key,   # API kimlik doğrulama
            model="text-embedding-ada-002"   # Embedding modeli
        )
        self.veritabani = None  # FAISS nesnesi (oluşturulana kadar None)

    def olustur_ve_kaydet(
        self,
        parcalar: List[Document],
        kayit_dizini: str = "./faiss_index"
    ) -> None:
        """
        Parçaları vektöre çevirir, FAISS'e yükler ve diske kaydeder.

        Args:
            parcalar      : MetinBolucu'dan gelen Document parçaları
            kayit_dizini  : FAISS index dosyasının kaydedileceği klasör
        """
        logger.info(f"{len(parcalar)} parça için embedding oluşturuluyor...")

        # from_documents: her Document'ı embedding_modeli ile vektöre çevirir
        # ve tüm vektörleri FAISS index'ine ekler → bu adım API çağrısı yapar!
        self.veritabani = FAISS.from_documents(
            parcalar,               # Vektörleştirilecek parçalar
            self.embedding_modeli   # Hangi embedding modelini kullan
        )

        # FAISS index'ini diske yaz (index.faiss + index.pkl dosyaları)
        self.veritabani.save_local(kayit_dizini)
        logger.info(f"Vektör veritabanı kaydedildi: {kayit_dizini}")

    def yukle(self, kayit_dizini: str = "./faiss_index") -> None:
        """
        Daha önce kaydedilmiş FAISS index'ini diskten yükler.
        Yeni belge yoksa her seferinde embedding üretmekten kaçınır.

        Args:
            kayit_dizini: Kaydedilmiş FAISS index klasörü
        """
        logger.info(f"Vektör veritabanı yükleniyor: {kayit_dizini}")

        # allow_dangerous_deserialization=True: pickle dosyası güvenli kabul edilir
        # (Sadece kendi oluşturduğunuz index'ler için kullanın!)
        self.veritabani = FAISS.load_local(
            kayit_dizini,
            self.embedding_modeli,
            allow_dangerous_deserialization=True
        )
        logger.info("Vektör veritabanı başarıyla yüklendi.")

    def benzer_ara(self, sorgu: str, k: int = 4) -> List[Document]:
        """
        Sorguya en benzer k belge parçasını döndürür.
        (Ham arama - LLM kullanmadan önce test için kullanışlıdır)

        Args:
            sorgu : Aranacak metin / kullanıcı sorusu
            k     : Döndürülecek maksimum parça sayısı

        Returns:
            En benzer k Document nesnesi listesi
        """
        if not self.veritabani:
            raise ValueError("Önce olustur_ve_kaydet() veya yukle() çağrılmalı!")

        # similarity_search: sorguyu vektöre çevirir, L2/kosinüs mesafesiyle arar
        sonuclar = self.veritabani.similarity_search(sorgu, k=k)
        logger.info(f"'{sorgu[:50]}...' için {len(sonuclar)} sonuç bulundu.")
        return sonuclar


# ─────────────────────────────────────────────────────────
#  4. RAG ZİNCİRİ (RETRIEVAL + GENERATION)
# ─────────────────────────────────────────────────────────
class RAGZinciri:
    """
    Vektör araması (retrieval) ile dil modeli üretimini (generation)
    birleştiren ana RAG sınıfı.
    """

    # Türkçe sistem promptu: LLM'e nasıl davranması gerektiğini söyler
    SISTEM_PROMPTU = """Sen yardımcı bir asistansın. Aşağıdaki bağlam bilgilerini
kullanarak soruyu Türkçe olarak yanıtla. Eğer bağlamda yeterli bilgi yoksa,
'Bu bilgi bağlamda yer almıyor.' diyerek dürüstçe belirt.

Bağlam:
{context}

Soru: {question}

Yanıt:"""

    def __init__(
        self,
        vektor_depo: VektorDepo,            # Arama yapılacak vektör DB
        model_adi: str = "gpt-4o-mini",     # Kullanılacak LLM modeli
        openai_api_key: str = "",           # OpenAI API anahtarı
        geri_donus_k: int = 4,              # Retrieval'da kaç parça dönsün
        sicaklik: float = 0.0               # 0=belirleyici, 1=yaratıcı
    ):
        self.vektor_depo = vektor_depo

        # ChatOpenAI: OpenAI chat completion endpoint'ine bağlanır
        self.llm = ChatOpenAI(
            model_name=model_adi,           # GPT modeli seçimi
            temperature=sicaklik,           # Yanıt rastgeleliği (0=tutarlı)
            openai_api_key=openai_api_key   # API kimlik doğrulama
        )

        # PromptTemplate: {context} ve {question} yer tutucularını doldurur
        self.prompt = PromptTemplate(
            template=self.SISTEM_PROMPTU,
            input_variables=["context", "question"]  # Şablondaki değişkenler
        )

        # Retriever: vektör DB'den benzer parçaları getiren nesne
        self.retriever = vektor_depo.veritabani.as_retriever(
            search_type="similarity",           # Benzerlik araması türü
            search_kwargs={"k": geri_donus_k}   # Kaç sonuç dönsün
        )

        # RetrievalQA zinciri: retriever + prompt + llm'i birleştirir
        self.zincir = RetrievalQA.from_chain_type(
            llm=self.llm,                       # Yanıt üretecek LLM
            chain_type="stuff",                 # 'stuff': tüm parçaları tek prompt'a doldur
            # Alternatifler: 'map_reduce' (uzun bağlam), 'refine' (iteratif iyileştirme)
            retriever=self.retriever,           # Belge getirme nesnesi
            return_source_documents=True,       # Kaynak parçaları da döndür
            chain_type_kwargs={"prompt": self.prompt}  # Özel prompt şablonu
        )

    def sor(self, soru: str) -> dict:
        """
        Kullanıcı sorusunu RAG zinciriyle yanıtlar.

        Akış:
          soru → embedding → FAISS arama → bağlam parçaları
               → prompt şablonu → LLM → yanıt

        Args:
            soru: Kullanıcının doğal dilde sorusu

        Returns:
            {
              'result'           : LLM'in ürettiği yanıt metni,
              'source_documents' : Yanıtın dayandığı Document listesi
            }
        """
        logger.info(f"Soru: {soru}")

        # invoke(): zinciri çalıştırır, {'query': soru} dict olarak iletilir
        yanit = self.zincir.invoke({"query": soru})

        # Kaynak belgelerin meta verilerini logla
        for i, doc in enumerate(yanit.get("source_documents", [])):
            logger.info(
                f"  Kaynak {i+1}: {doc.metadata.get('source', 'Bilinmiyor')} "
                f"| Sayfa: {doc.metadata.get('page', '-')}"
            )

        return yanit


# ─────────────────────────────────────────────────────────
#  5. ANA RAG PIPELINE SINIFI (Tüm adımları birleştirir)
# ─────────────────────────────────────────────────────────
class RAGPipeline:
    """
    Yükleme → Bölme → Vektörleştirme → Yanıt üretme
    adımlarını tek bir arayüzde birleştiren üst düzey sınıf.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,  # None ise env değişkenine bakılır
        parca_boyutu: int = 1000,              # Metin parça boyutu
        parca_otusmesi: int = 200,             # Parçalar arası örtüşme
        geri_donus_k: int = 4,                 # Retrieval sonuç sayısı
        model_adi: str = "gpt-4o-mini",        # LLM model adı
        index_dizini: str = "./faiss_index"    # FAISS kayıt/yükleme dizini
    ):
        # API anahtarı: parametre > ortam değişkeni > hata
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenAI API anahtarı gerekli! "
                "openai_api_key parametresi veya OPENAI_API_KEY env değişkeni set edin."
            )

        self.index_dizini = index_dizini     # FAISS index klasörü
        self.model_adi = model_adi           # LLM model adı
        self.geri_donus_k = geri_donus_k     # Retrieval k değeri

        # Alt bileşenleri oluştur
        self.yukleyici = BelgeYukleyici()                          # Belge yükleme
        self.bolucu = MetinBolucu(parca_boyutu, parca_otusmesi)   # Metin bölme
        self.vektor_depo = VektorDepo(self.api_key)               # Vektör DB
        self.rag_zinciri: Optional[RAGZinciri] = None             # RAG zinciri (sonra oluşur)

    def belgelerden_index_olustur(self, belge_yolu: str) -> None:
        """
        Belge dosyasından/klasöründen sıfırdan FAISS index oluşturur.
        (İlk kurulum veya yeni belgeler eklendiğinde çalıştırılır)

        Args:
            belge_yolu: PDF dosyası, TXT dosyası veya klasör yolu
        """
        yol = Path(belge_yolu)  # pathlib.Path nesnesi (platform bağımsız)

        # Dosya türüne göre doğru yükleyiciyi seç
        if yol.is_dir():
            # Klasör: içindeki tüm TXT dosyalarını yükle
            belgeler = self.yukleyici.klasor_yukle(str(yol))
        elif yol.suffix.lower() == ".pdf":
            # PDF dosyası
            belgeler = self.yukleyici.pdf_yukle(str(yol))
        elif yol.suffix.lower() == ".txt":
            # Düz metin dosyası
            belgeler = self.yukleyici.txt_yukle(str(yol))
        else:
            # Desteklenmeyen format → hata fırlat
            raise ValueError(f"Desteklenmeyen dosya türü: {yol.suffix}")

        # Yüklenen belgeleri parçalara böl
        parcalar = self.bolucu.parcala(belgeler)

        # Parçaları vektörleştir ve FAISS'e kaydet
        self.vektor_depo.olustur_ve_kaydet(parcalar, self.index_dizini)

        # RAG zincirini hazırla
        self._rag_zinciri_olustur()
        logger.info("Pipeline hazır! Soru sorabilirsiniz.")

    def mevcut_indexten_yukle(self) -> None:
        """
        Daha önce kaydedilmiş FAISS index'ini yükler.
        (Her seferinde embedding üretmekten kaçınır → hızlı başlangıç)
        """
        self.vektor_depo.yukle(self.index_dizini)  # Diskten FAISS'i yükle
        self._rag_zinciri_olustur()                 # RAG zincirini hazırla
        logger.info("Mevcut index'ten pipeline yüklendi.")

    def _rag_zinciri_olustur(self) -> None:
        """
        Dahili: Vektör DB hazır olduktan sonra RAG zincirini oluşturur.
        (Bu metod doğrudan kullanıcı tarafından çağrılmaz)
        """
        self.rag_zinciri = RAGZinciri(
            vektor_depo=self.vektor_depo,
            model_adi=self.model_adi,
            openai_api_key=self.api_key,
            geri_donus_k=self.geri_donus_k
        )

    def sor(self, soru: str) -> str:
        """
        Pipeline'a soru sorar ve yanıt metnini döndürür.

        Args:
            soru: Doğal dilde kullanıcı sorusu

        Returns:
            LLM'in ürettiği yanıt metni (string)

        Raises:
            RuntimeError: Pipeline henüz hazır değilse
        """
        if not self.rag_zinciri:
            raise RuntimeError(
                "Pipeline hazır değil! Önce belgelerden_index_olustur() "
                "veya mevcut_indexten_yukle() çağırın."
            )

        yanit = self.rag_zinciri.sor(soru)    # RAG zincirini çalıştır
        return yanit["result"]                 # Sadece metin yanıtı döndür

    def sor_detayli(self, soru: str) -> dict:
        """
        Yanıt metninin yanı sıra kaynak belgeleri de döndürür.
        Şeffaflık ve debug için kullanışlıdır.

        Returns:
            {
              'yanit'    : LLM yanıtı,
              'kaynaklar': [{'icerik': ..., 'kaynak': ..., 'sayfa': ...}, ...]
            }
        """
        if not self.rag_zinciri:
            raise RuntimeError("Pipeline hazır değil!")

        ham_yanit = self.rag_zinciri.sor(soru)   # Ham zincir çıktısı

        # Kaynak belgelerden okunabilir bilgi çıkar
        kaynaklar = []
        for doc in ham_yanit.get("source_documents", []):
            kaynaklar.append({
                "icerik": doc.page_content[:200] + "...",        # İlk 200 karakter
                "kaynak": doc.metadata.get("source", "Bilinmiyor"),  # Dosya adı
                "sayfa" : doc.metadata.get("page", "-")          # Sayfa numarası
            })

        return {
            "yanit"    : ham_yanit["result"],  # LLM yanıtı
            "kaynaklar": kaynaklar             # Kaynak listesi
        }


# ─────────────────────────────────────────────────────────
#  6. DEMO / TEST BLOĞU
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Bu blok sadece dosya doğrudan çalıştırıldığında aktif olur.
    (import edildiğinde çalışmaz → __name__ == "__main__" kontrolü)
    """

    # ── Konfigürasyon ─────────────────────────────────────
    API_KEY       = os.getenv("OPENAI_API_KEY", "sk-...")   # API anahtarı
    BELGE_YOLU    = "./belgeler"          # İndexlenecek dosya/klasör
    INDEX_DIZINI  = "./faiss_index"       # FAISS kayıt klasörü
    MODEL         = "gpt-4o-mini"         # Kullanılacak LLM modeli

    # ── Pipeline'ı başlat ─────────────────────────────────
    pipeline = RAGPipeline(
        openai_api_key=API_KEY,
        parca_boyutu=1000,       # Her parça maks. 1000 karakter
        parca_otusmesi=200,      # Parçalar 200 karakter örtüşür
        geri_donus_k=4,          # Her sorguda 4 parça getir
        model_adi=MODEL,
        index_dizini=INDEX_DIZINI
    )

    # ── Index oluştur veya mevcut olanı yükle ─────────────
    if Path(INDEX_DIZINI).exists():
        # Daha önce oluşturulmuş index varsa yükle (hızlı)
        pipeline.mevcut_indexten_yukle()
    else:
        # İlk çalıştırma: belgeleri oku, parçala, vektörleştir
        pipeline.belgelerden_index_olustur(BELGE_YOLU)

    # ── Soru-Yanıt döngüsü ────────────────────────────────
    sorular = [
        "Bu belgeler ne hakkında?",
        "Ana konular nelerdir?",
        "Önemli sonuçlar neler?",
    ]

    print("\n" + "="*60)
    print("  RAG SİSTEMİ HAZIR - SORU-YANIT BAŞLIYOR")
    print("="*60)

    for soru in sorular:
        print(f"\n❓ Soru : {soru}")

        # Detaylı yanıt (kaynaklar dahil)
        sonuc = pipeline.sor_detayli(soru)

        print(f"💬 Yanıt: {sonuc['yanit']}")
        print("📚 Kaynaklar:")
        for k in sonuc["kaynaklar"]:
            # her kaynak için: dosya adı + sayfa numarası + içerik önizlemesi
            print(f"   • [{k['kaynak']} | Sayfa {k['sayfa']}] {k['icerik']}")

    print("\n" + "="*60)
    print("  İnteraktif Mod (çıkmak için 'q' yazın)")
    print("="*60)

    # Kullanıcıdan dinamik soru al
    while True:
        kullanici_sorusu = input("\nSorunuzu yazın: ").strip()

        if kullanici_sorusu.lower() in ("q", "quit", "çıkış", "exit"):
            print("Çıkılıyor...")
            break  # Döngüden çık

        if not kullanici_sorusu:
            continue  # Boş giriş → tekrar sor

        # Pipeline'a sor ve yanıtı yazdır
        yanit = pipeline.sor(kullanici_sorusu)
        print(f"\n💬 Yanıt: {yanit}")
