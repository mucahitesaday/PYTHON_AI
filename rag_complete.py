"""
====================================================================
  RAG (Retrieval-Augmented Generation) — A'dan Z'ye Tam Mimari
  ====================================================================
  Bu dosya, RAG sistemini sıfırdan uçtan uca kurar.

  RAG Akışı:
    1. Kullanıcı bir soru sorar
    2. Belgeler parçalara (chunk) bölünür
    3. Her parça sayısal vektöre dönüştürülür (embedding)
    4. Vektörler bir veritabanına kaydedilir
    5. Kullanıcının sorusu da vektöre çevrilir
    6. En benzer parçalar veritabanından bulunur (retrieval)
    7. Bulunan parçalar + soru → LLM'e gönderilir
    8. LLM bağlama dayalı yanıt üretir (generation)

  Kapsanan tüm konular:
    1.  Konfigürasyon    — config.dataclass + .env + YAML dosyası
    2.  Belge yükleme    — PDF, TXT, DOCX, CSV, HTML, MD, JSON
    3.  Chunking         — recursive, token, character stratejileri
    4.  Embedding        — OpenAI, HuggingFace, Ollama seçenekleri
    5.  Vektör deposu    — FAISS, Chroma, InMemory
    6.  Retrieval        — similarity, MMR, threshold
    7.  Reranker         — Cross-encoder ile sıralama iyileştirme
    8.  Sorgu dönüşümü   — HyDE, Multi-Query
    9.  Konuşma hafızası — Buffer, Window (geçmişi hatırlama)
    10. Streaming        — Token token yanıt üretme
    11. Değerlendirme    — faithfulness, relevancy, accuracy
    12. FastAPI servisi  — REST API ile sorgulama
    13. Gradio arayüzü   — Web tabanlı chat + dosya yükleme
    14. CLI uygulaması   — Komut satırından kullanım

  Kurulum:
    pip install langchain langchain-community langchain-openai
    pip install langchain-huggingface langchain-chroma
    pip install faiss-cpu tiktoken pypdf python-docx
    pip install sentence-transformers chromadb
    pip install fastapi uvicorn gradio python-multipart
    pip install pydantic-settings pyyaml
    pip install unstructured markdown beautifulsoup4 lxml
    pip install python-dotenv
====================================================================
"""

# ── __future__ import'u: Python'un eski sürümlerinde yeni özellikleri
#    kullanabilmek için. annotations → tip ipuçlarının ertelenmesini sağlar.
from __future__ import annotations

import json      # JSON okuma/yazma (config, eval sonuçları)
import logging   # Log mesajları (hata/uyarı/bilgi seviyeleri)
import os        # İşletim sistemi: dosya/dizin işlemleri, env değişkenleri
import sys       # Sistem parametreleri (çıkış kodları vb.)
import tempfile  # Geçici dosya oluşturma (API dosya yükleme için)
from abc import ABC, abstractmethod  # Soyut sınıflar → zorunlu metodlar
from dataclasses import dataclass, field  # @dataclass: veri taşıyan sınıflar
from enum import Enum                # Enum: sabit değer grupları
from pathlib import Path             # Dosya yollarını nesne olarak yönetir
from typing import Any, Callable, Dict, Generator, List, Optional, Type, Union

import yaml     # YAML dosyası okuma (konfigürasyon için)
from dotenv import load_dotenv  # .env dosyasındaki değişkenleri yükler

# ── LangChain: LLM uygulamaları için popüler framework ──────────
# LangChain, zincirleme LLM çağrıları, belge yönetimi, vektör araması gibi
# RAG için gerekli tüm araçları sağlar.
from langchain_ollama import ChatOllama  # Ollama (yerel LLM)
from langchain_community.document_loaders import (
    CSVLoader,                    # CSV dosyası yükleyici
    DirectoryLoader,              # Tüm klasörü tarayan yükleyici
    Docx2txtLoader,               # Word (.docx) dosyası yükleyici
    JSONLoader,                   # JSON dosyası yükleyici
    PyPDFLoader,                  # PDF dosyası yükleyici
    TextLoader,                   # Düz metin (.txt) yükleyici
    UnstructuredHTMLLoader,       # HTML sayfası yükleyici
    UnstructuredMarkdownLoader,   # Markdown (.md) yükleyici
)
from langchain_ollama import OllamaEmbeddings  # Ollama embedding
from langchain_community.vectorstores import Chroma, FAISS  # Vektör DB'leri
from langchain_core.documents import Document  # LangChain temel belge sınıfı
from langchain_core.language_models import BaseLanguageModel  # LLM arayüzü
from langchain_core.messages import AIMessage, HumanMessage  # Sohbet mesajları
from langchain_core.output_parsers import StrOutputParser  # String çıktı ayrıştırıcı
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
# PromptTemplate: LLM'e gönderilecek prompt'u şablonlaştırır
# ChatPromptTemplate: Sohbet tabanlı prompt'lar için
from langchain_core.runnables import Runnable, RunnablePassthrough  # LCEL zincirleri
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever  # Vektör DB arayüzü
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace embedding
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI LLM + embedding
from langchain_text_splitters import (
    CharacterTextSplitter,        # Karakter bazlı metin bölücü
    RecursiveCharacterTextSplitter,  # Özyinelemeli metin bölücü (önerilen)
    TokenTextSplitter,            # Token (kelime parçası) bazlı bölücü
)

# .env dosyasını yükle → içindeki OPENAI_API_KEY vb. değişkenler okunur
load_dotenv()

# Log ayarları: INFO seviyesi ve üstü mesajları göster
# Format: "2024-01-01 12:00:00 [INFO] modul: mesaj"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag")  # Bu modüle özel logger


# ══════════════════════════════════════════════════════════════════════
#  1. KONFIGÜRASYON (Configuration)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: Tüm RAG ayarlarını tek bir yerden yönetmek.
#  Enum'lar: Kullanıcının seçim yapmasını kolaylaştırır ve hataları önler.
#  @dataclass: Python'un otomatik __init__, __repr__ vb. ürettiği veri sınıfı.
#  Config; YAML dosyasından, dict'ten veya doğrudan parametreyle oluşturulabilir.
# ══════════════════════════════════════════════════════════════════════


class EmbeddingProvider(str, Enum):
    """
    Hangi embedding sağlayıcısının kullanılacağını belirler.
    Embedding: Metni sayısal vektöre dönüştürme işlemi.
      OPENAI     → text-embedding-ada-002 (ücretli, kaliteli)
      HUGGINGFACE → all-MiniLM-L6-v2 (ücretsiz, yerel)
      OLLAMA     → nomic-embed-text (ücretsiz, yerel)
    """
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class VectorStoreType(str, Enum):
    """
    Vektör veritabanı türü.
    Vektör DB: Embedding'leri saklar ve benzerlik araması yapar.
      FAISS    → Facebook'un vektör arama kütüphanesi (bellek içi, hızlı)
      CHROMA   → ChromaDB (diske kaydeder, sorgulanabilir)
      IN_MEMORY → FAISS'in bellekte tutulan hali (geçici)
    """
    FAISS = "faiss"
    CHROMA = "chroma"
    IN_MEMORY = "in_memory"


class ChunkStrategy(str, Enum):
    """
    Metin bölme stratejisi.
    Chunking: Uzun belgeleri küçük parçalara bölme.
      RECURSIVE  → Önce paragraf, sonra cümle, sonra kelime (önerilen)
      TOKEN      → Token sayısına göre böler (LLM context sınırı için ideal)
      CHARACTER  → Sabit karakter sayısına göre böler
    """
    RECURSIVE = "recursive"
    TOKEN = "token"
    CHARACTER = "character"


class RetrievalStrategy(str, Enum):
    """
    Vektör veritabanından belge getirme stratejisi.
      SIMILARITY              → En benzer k belge (standart)
      MMR                     → Çeşitliliği de dikkate al (max marginal relevance)
      SIMILARITY_SCORE_THRESHOLD → Skor eşiğinin üstündeki belgeler
    """
    SIMILARITY = "similarity"
    MMR = "mmr"
    SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"


class LLMProvider(str, Enum):
    """
    Büyük dil modeli sağlayıcısı.
      OPENAI      → GPT-4, GPT-3.5 (ücretli, API gerekli)
      OLLAMA      → Yerel modeller (Llama 3, Mistral vb.) (ücretsiz)
      HUGGINGFACE → HuggingFace modelleri (transformers ile yerel çalışır)
    """
    OPENAI = "openai"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


@dataclass
class RAGConfig:
    """
    Merkezi konfigürasyon nesnesi.
    Tüm RAG sistemi ayarları bu sınıfta toplanır.
    Varsayılan değerler atanmıştır, istenirse değiştirilebilir.

    Kullanım:
      config = RAGConfig(llm_model="gpt-4o", chunk_size=500)
      config = RAGConfig.from_yaml("config.yaml")
      config = RAGConfig.from_dict({"llm_model": "gpt-4o"})
    """

    # ── LLM Ayarları ─────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.OPENAI
    # Hangi LLM sağlayıcısı? (openai/ollama)
    llm_model: str = "gpt-4o-mini"
    # Model adı: OpenAI için "gpt-4o-mini", Ollama için "llama3"
    llm_temperature: float = 0.0
    # Sıcaklık: 0 = tutarlı/deterministik, 1 = yaratıcı/rastgele
    llm_max_tokens: Optional[int] = None
    # Maksimum yanıt token sayısı (None = model limiti)
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # OpenAI API anahtarı: önce parametre, yoksa env değişkeni
    ollama_base_url: str = "http://localhost:11434"
    # Ollama sunucu adresi (varsayılan: localhost)

    # ── HuggingFace LLM Ayarları ──────────────────────────────────
    hf_llm_model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    # HuggingFace'den indirilecek LLM model adı (açık erişimli)
    hf_device: str = "auto"
    # Cihaz: "auto" (GPU varsa GPU, yoksa CPU), "cuda", "cpu"
    hf_max_new_tokens: int = 512
    # Her yanıt için maksimum yeni token sayısı
    hf_quantize: bool = True
    # 4-bit quantization (bitsandbytes) — VRAM tasarrufu için
    hf_load_in_8bit: bool = False
    # 8-bit quantization — daha kaliteli ama daha fazla VRAM

    # ── Embedding Ayarları ───────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    # Embedding sağlayıcısı
    embedding_model: str = "text-embedding-ada-002"
    # OpenAI embedding modeli (1536 boyutlu vektör üretir)
    huggingface_embedding_model: str = "all-MiniLM-L6-v2"
    # HuggingFace modeli (384 boyutlu, hızlı ve hafif)
    ollama_embedding_model: str = "nomic-embed-text"
    # Ollama embedding modeli

    # ── Vektör Deposu Ayarları ───────────────────────────────────
    vector_store: VectorStoreType = VectorStoreType.FAISS
    # Kullanılacak vektör veritabanı türü
    index_directory: str = "./rag_index"
    # Index dosyalarının kaydedileceği klasör

    # ── Chunking (Metin Bölme) Ayarları ──────────────────────────
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    # Hangi bölme stratejisi kullanılsın?
    chunk_size: int = 1000
    # Her parçanın maksimum karakter sayısı
    chunk_overlap: int = 200
    # Ardışık parçalar arası örtüşme (bağlam kaybını önler)

    # ── Retrieval (Belge Getirme) Ayarları ───────────────────────
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY
    # Arama stratejisi
    retrieval_k: int = 4
    # Kaç adet en benzer belge getirilecek?
    retrieval_fetch_k: int = 20
    # MMR için: önce kaç belge getirilsin, sonra çeşitlendirilsin?
    retrieval_lambda_mult: float = 0.5
    # MMR çeşitlilik parametresi: 0=tam çeşitlilik, 1=tam benzerlik
    retrieval_score_threshold: float = 0.5
    # Skor eşiği: sadece bu skorun üstündeki belgeler getirilir

    # ── Reranking (Yeniden Sıralama) Ayarları ────────────────────
    use_reranker: bool = False
    # Reranker kullanılsın mı? (Cross-encoder ile hassas sıralama)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Kullanılacak reranker modeli
    reranker_top_k: int = 3
    # Reranker'dan sonra kaç belge tutulsun?

    # ── Sorgu Dönüşümü Ayarları ──────────────────────────────────
    use_hyde: bool = False
    # HyDE: Önce hayali cevap üret, sonra onunla ara (daha iyi eşleşme)
    use_multi_query: bool = False
    # Multi-Query: Sorunun farklı versiyonlarını üret, hepsiyle ara
    use_query_decomposition: bool = False
    # Query Decomposition: Karmaşık soruyu alt sorulara böl (ileri seviye)
    multi_query_count: int = 3
    # Multi-Query'de kaç farklı versiyon üretilsin?

    # ── Hafıza (Conversation Memory) Ayarları ────────────────────
    use_memory: bool = False
    # Konuşma geçmişi tutulsun mu? (çok turlu sohbet için)
    memory_type: str = "buffer"
    # Hafıza türü: "buffer" (tümü) | "summary" (özet) | "window" (son N mesaj)
    memory_window: int = 5
    # Window hafızada son kaç mesaj tutulsun?

    # ── Değerlendirme Ayarları ───────────────────────────────────
    eval_llm_model: str = "gpt-4o-mini"
    # Değerlendirme için kullanılacak LLM (jüri modeli)

    # ── Genel Ayarlar ────────────────────────────────────────────
    verbose: bool = False
    # Ayrıntılı log çıktısı (debug için)

    # Varsayılan sistem promptu: LLM'in nasıl davranacağını belirler
    # {context} → vektör DB'den gelen belge parçaları
    # {question} → kullanıcının sorusu
    system_prompt: str = """Sen yardımcı bir asistansın. Aşağıdaki bağlam bilgilerini
kullanarak soruyu Türkçe olarak yanıtla. Eğer bağlamda yeterli bilgi yoksa,
'Bu bilgi bağlamda yer almıyor.' diyerek dürüstçe belirt.

Bağlam:
{context}

Soru: {question}

Yanıt:"""

    # ── Alternatif kurucu metodlar ───────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> RAGConfig:
        """
        YAML dosyasından config oluşturur.
        Örnek YAML:
          llm_model: gpt-4o
          chunk_size: 500
          use_reranker: true
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)  # YAML → Python dict
        # Sadece RAGConfig'da tanımlı alanları al (fazlalıkları at)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    @classmethod
    def from_dict(cls, data: dict) -> RAGConfig:
        """Sözlükten config oluşturur. (API'den gelen veri için)"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# ══════════════════════════════════════════════════════════════════════
#  2. BELGE YÜKLEYİCİ (Document Loader)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: Farklı formatlardaki belgeleri LangChain Document nesnesine çevirmek.
#  Document: page_content (metin) + metadata (kaynak, sayfa no vb.) içerir.
#  Factory Pattern: Hangi format geldiyse ona göre doğru yükleyici seçilir.
#  Desteklenen formatlar: PDF, TXT, DOCX, CSV, HTML, MD, JSON.
# ══════════════════════════════════════════════════════════════════════


class DocumentLoaderFactory:
    """
    Fabrika tasarım deseni: Her dosya uzantısına göre uygun yükleyiciyi
    seçer ve belgeyi LangChain Document listesine dönüştürür.

    Kullanım:
      docs = DocumentLoaderFactory.load("belge.pdf")
      docs = DocumentLoaderFactory.load("klasor/")
    """

    # Dosya uzantısı → yükleyici fonksiyonu eşlemesi
    # Her uzantı için bir lambda (anonim fonksiyon) tanımlı
    loaders: Dict[str, Callable[[str], Any]] = {
        ".pdf": lambda p: PyPDFLoader(p),                          # PDF → her sayfa ayrı Document
        ".txt": lambda p: TextLoader(p, encoding="utf-8"),         # TXT → tek Document
        ".docx": lambda p: Docx2txtLoader(p),                      # Word → tek Document
        ".csv": lambda p: CSVLoader(p),                            # CSV → her satır ayrı Document
        ".html": lambda p: UnstructuredHTMLLoader(p),               # HTML → metin olarak
        ".htm": lambda p: UnstructuredHTMLLoader(p),               # .htm de aynı
        ".md": lambda p: UnstructuredMarkdownLoader(p),            # Markdown → metin olarak
        ".json": lambda p: JSONLoader(p, jq_schema=".", text_content=False),  # JSON
    }

    @classmethod
    def load(cls, path: str) -> List[Document]:
        """
        Ana yükleme metodu. Dosya veya klasör yolunu alır.
        - Klasör ise içindeki tüm desteklenen dosyaları tara
        - Dosya ise uzantısına bak, uygun yükleyiciyi seç
        """
        path_obj = Path(path)       # String'i Path nesnesine çevir
        suffix = path_obj.suffix.lower()  # Uzantıyı al (.pdf, .txt vb.)

        if path_obj.is_dir():
            # Klasör → recursive tarama
            return cls._load_directory(path_obj)

        # Uzantıya göre yükleyici bul
        loader_fn = cls.loaders.get(suffix)
        if not loader_fn:
            raise ValueError(
                f"Desteklenmeyen dosya türü: {suffix}. "
                f"Desteklenenler: {list(cls.loaders.keys())}"
            )

        logger.info(f"Yükleniyor: {path}")
        loader = loader_fn(path)    # Yükleyici nesnesini oluştur
        docs = loader.load()         # Belgeyi yükle → Document listesi
        logger.info(f"{len(docs)} belge yüklendi: {path}")
        return docs

    @classmethod
    def _load_directory(cls, directory: Path, glob_pattern: str = "**/*") -> List[Document]:
        """
        Bir klasördeki tüm desteklenen dosyaları tarar.
        Her uzantı için ayrı ayrı DirectoryLoader çalıştırır.
        """
        logger.info(f"Klasör taranıyor: {directory} /**")
        all_docs: List[Document] = []

        # Her dosya uzantısı için ayrı loader çalıştır
        for ext, loader_fn in cls.loaders.items():
            pattern = f"**/*{ext}"  # Ör: "**/*.pdf", "**/*.txt"
            loader = DirectoryLoader(
                str(directory),
                glob=pattern,
                loader_cls=type(loader_fn("dummy")),  # Yükleyici sınıfını belirle
                use_multithreading=True,  # Çoklu iş parçacığı → hızlı tarama
            )
            try:
                docs = loader.load()
                all_docs.extend(docs)
                if docs:
                    logger.info(f"  {pattern}: {len(docs)} belge")
            except Exception as e:
                # Hata olursa o formatı atla, devam et
                logger.warning(f"  {pattern}: Atlanıyor — {e}")

        logger.info(f"Toplam {len(all_docs)} belge yüklendi.")
        return all_docs


# ══════════════════════════════════════════════════════════════════════
#  3. METİN BÖLME (Chunking)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: Uzun belgeleri LLM'in context penceresine sığacak parçalara bölmek.
#  Neden önemli?
#    - LLM'lerin sınırlı context penceresi vardır (8K-128K token)
#    - Küçük parçalar daha hassas arama yapmayı sağlar
#    - Her parça bağımsız olarak vektörleştirilir
#
#  Stratejiler:
#    RECURSIVE: Önce paragraf (\n\n), sonra satır (\n), sonra kelime (' ')
#               → en anlamlı bölünmeyi sağlar (önerilen)
#    TOKEN:     LLM tokenizer'ına göre böler (context sınırı için ideal)
#    CHARACTER: Sabit karakter sayısına göre böler (basit ama anlamsız)
# ══════════════════════════════════════════════════════════════════════


class Chunker(ABC):
    """
    Tüm chunker'ların temel sınıfı.
    ABC → abstractmethod: Alt sınıflar split() metodunu tanımlamak ZORUNDADIR.
    """
    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]: ...


class RecursiveChunker(Chunker):
    """
    Özyinelemeli metin bölücü.
    Bölme sırası: paragraf → satır → cümle → kelime → karakter
    Önce en büyük yapıyı (paragraf) bölmeyi dener, olmazsa küçültür.
    Bu sayede anlam bütünlüğü en iyi korunur.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,       # Maksimum karakter sayısı
            chunk_overlap=chunk_overlap, # Parçalar arası örtüşme (bağlam koruma)
            length_function=len,         # Uzunluk ölçümü: karakter sayısı
            separators=["\n\n", "\n", ".", " ", ""],  # Bölme öncelik sırası
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """Belge listesini parçalara böler."""
        return self.splitter.split_documents(documents)


class TokenChunker(Chunker):
    """
    Token bazlı bölücü.
    Token: Kelimelerin/hecelerin sayısal temsili.
    LLM'ler token ile çalıştığı için context sınırına göre bölmek idealdir.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,       # Maksimum token sayısı
            chunk_overlap=chunk_overlap, # Token bazlı örtüşme
        )

    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class CharacterChunker(Chunker):
    """
    Karakter bazlı bölücü (en basiti).
    Sabit sayıda karaktere göre böler. Anlam bütünlüğü zayıftır.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n",  # Önce paragraf sonlarından böl
        )

    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class ChunkerFactory:
    """
    Chunker fabrikası: Hangi strateji seçildiyse onu oluşturur.
    Token stratejisi: chunk_size/4 (1 token ≈ 4 karakter).
    """
    @staticmethod
    def create(strategy: ChunkStrategy, chunk_size: int, chunk_overlap: int) -> Chunker:
        # Strateji → sınıf eşlemesi
        mapping: Dict[ChunkStrategy, Type[Chunker]] = {
            ChunkStrategy.RECURSIVE: RecursiveChunker,
            ChunkStrategy.TOKEN: TokenChunker,
            ChunkStrategy.CHARACTER: CharacterChunker,
        }
        chunker_cls = mapping[strategy]  # Seçilen stratejinin sınıfını al

        if strategy == ChunkStrategy.TOKEN:
            # Token: karakter/4 (ortalama token boyutu)
            return chunker_cls(
                chunk_size=chunk_size // 4,
                chunk_overlap=chunk_overlap // 4
            )
        return chunker_cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )


# ══════════════════════════════════════════════════════════════════════
#  4. EMBEDDING SAĞLAYICILARI (Embedding Factory)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: Metni sayısal vektöre dönüştürmek.
#  Embedding: "Köpek" ve "Kedi" benzer vektörlere → ikisi de hayvan.
#              "Köpek" ve "Araba" farklı vektörlere → alakasız kavramlar.
#  Bu benzerlik vektör aramasının temelidir.
#
#  3 seçenek:
#    OpenAI:    text-embedding-ada-002 (1536 boyut, yüksek kalite, ücretli)
#    HuggingFace: all-MiniLM-L6-v2 (384 boyut, hızlı, ücretsiz)
#    Ollama:    nomic-embed-text (yerel, ücretsiz)
# ══════════════════════════════════════════════════════════════════════


class EmbeddingFactory:
    """
    Embedding sağlayıcısını oluşturur.
    Kullanıcının config'de seçtiği provider'a göre doğru sınıfı döndürür.
    """
    @staticmethod
    def create(config: RAGConfig):
        if config.embedding_provider == EmbeddingProvider.OPENAI:
            # OpenAI: API anahtarı gerekli, en kaliteli embedding
            return OpenAIEmbeddings(
                openai_api_key=config.openai_api_key,
                model=config.embedding_model,  # text-embedding-ada-002
            )
        elif config.embedding_provider == EmbeddingProvider.HUGGINGFACE:
            # HuggingFace: tamamen yerel çalışır, internet gerekmez
            return HuggingFaceEmbeddings(
                model_name=config.huggingface_embedding_model,  # all-MiniLM-L6-v2
            )
        elif config.embedding_provider == EmbeddingProvider.OLLAMA:
            # Ollama: yerel LLM sunucusu üzerinden embedding
            return OllamaEmbeddings(
                model=config.ollama_embedding_model,  # nomic-embed-text
                base_url=config.ollama_base_url,
            )
        raise ValueError(f"Bilinmeyen embedding provider: {config.embedding_provider}")


# ══════════════════════════════════════════════════════════════════════
#  5. VEKTÖR DEPOSU (Vector Store)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: Embedding vektörlerini saklamak ve benzerlik araması yapmak.
#  Vektör deposu, RAG'ın "Retrieval" (arama) kısmının kalbidir.
#
#  3 seçenek:
#    FAISS:  Facebook AI'in kütüphanesi, hızlı, bellekte çalışır
#    Chroma: Diske kaydeder, sorgulanabilir, büyük veri için uygun
#    InMemory: Geçici, uygulama kapanınca kaybolur
# ══════════════════════════════════════════════════════════════════════


class VectorStoreFactory:
    """
    Vektör deposu oluşturma fabrikası.
    3 temel işlem:
      create()         → Boş bir vektör deposu oluşturur
      from_documents() → Belgeleri vektörleştirip depoya ekler
      load_local()     → Daha önce kaydedilmiş depoyu diskten yükler
    """
    @staticmethod
    def create(config: RAGConfig, embeddings) -> VectorStore:
        """Boş vektör deposu oluşturur (nadiren kullanılır)."""
        if config.vector_store == VectorStoreType.FAISS:
            return FAISS(embeddings)
        elif config.vector_store == VectorStoreType.CHROMA:
            return Chroma(embedding_function=embeddings)
        elif config.vector_store == VectorStoreType.IN_MEMORY:
            return FAISS(embeddings)  # InMemory = FAISS'in bellekteki hali
        raise ValueError(f"Bilinmeyen vector store: {config.vector_store}")

    @staticmethod
    def from_documents(
        config: RAGConfig, documents: List[Document], embeddings
    ) -> VectorStore:
        """
        Belgeleri vektörleştirir ve depoya ekler.
        En sık kullanılan metot: index oluşturma işleminin kalbi.
        """
        vs_type = config.vector_store
        if vs_type == VectorStoreType.FAISS:
            # FAISS.from_documents: her Document'ı embed eder ve index'e ekler
            return FAISS.from_documents(documents, embeddings)
        elif vs_type == VectorStoreType.CHROMA:
            # Chroma: belirtilen dizine kaydeder (otomatik persist)
            return Chroma.from_documents(
                documents, embeddings, persist_directory=config.index_directory
            )
        elif vs_type == VectorStoreType.IN_MEMORY:
            return FAISS.from_documents(documents, embeddings)
        raise ValueError(f"Bilinmeyen vector store: {vs_type}")

    @staticmethod
    def load_local(config: RAGConfig, embeddings) -> VectorStore:
        """
        Daha önce kaydedilmiş index'i diskten yükler.
        Her seferinde embedding üretmekten kaçınır (zaman kazancı).
        """
        vs_type = config.vector_store
        if vs_type == VectorStoreType.FAISS:
            # allow_dangerous_deserialization: pickle dosyası güvenli kabul edilir
            return FAISS.load_local(
                config.index_directory,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        elif vs_type == VectorStoreType.CHROMA:
            return Chroma(
                persist_directory=config.index_directory,
                embedding_function=embeddings,
            )
        raise ValueError(f"Yerel yükleme desteklenmiyor: {vs_type}")


# ══════════════════════════════════════════════════════════════════════
#  6. GELİŞMİŞ RETRIEVAL STRATEJİLERİ (Retriever)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: Vektör deposundan en alakalı belgeleri bulmak.
#  Retriever = soruyu vektöre çevir + vektör DB'de ara + sonuçları getir.
#
#  3 strateji:
#    SIMILARITY: En benzer k belge (standart kosinüs/L2 benzerliği)
#    MMR: Çeşitlilik + benzerlik dengesi (Max Marginal Relevance)
#    SIMILARITY_SCORE_THRESHOLD: Belli bir eşik skorun üstündekiler
# ══════════════════════════════════════════════════════════════════════


class RetrieverFactory:
    """
    Vektör deposundan bir retriever (belge getirici) oluşturur.
    Retriever daha sonra RAG zincirinde kullanılır.
    """
    @staticmethod
    def create(vectorstore: VectorStore, config: RAGConfig) -> VectorStoreRetriever:
        # Arama parametreleri: varsayılan k=4 (en benzer 4 belge)
        search_kwargs: Dict[str, Any] = {"k": config.retrieval_k}

        if config.retrieval_strategy == RetrievalStrategy.MMR:
            # MMR: Max Marginal Relevance
            # fetch_k: önce 20 belge getir, sonra içinden en iyi k'yı seç
            # lambda_mult: 0=çeşitlilik, 1=benzerlik
            search_kwargs["fetch_k"] = config.retrieval_fetch_k
            search_kwargs["lambda_mult"] = config.retrieval_lambda_mult

        elif config.retrieval_strategy == RetrievalStrategy.SIMILARITY_SCORE_THRESHOLD:
            # Skor eşiği: 0.5'in üstündeki belgeler getirilir
            search_kwargs["score_threshold"] = config.retrieval_score_threshold

        # as_retriever(): VectorStore → VectorStoreRetriever dönüşümü
        return vectorstore.as_retriever(
            search_type=config.retrieval_strategy.value,
            search_kwargs=search_kwargs,
        )


# ══════════════════════════════════════════════════════════════════════
#  7. RERANKER (Yeniden Sıralama)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: Vektör aramasından gelen sonuçları daha hassas bir modelle
#        yeniden sıralamak.
#
#  Neden gerekli?
#    Vektör araması (cosine similarity) hızlıdır ama bazen yanıltıcı olabilir.
#    Cross-encoder: sorgu + belge çiftini birlikte işler, çok daha doğru sıralar.
#
#  Cross-encoder: İki metni alır, aralarındaki ilişkiyi 0-1 arası puanlar.
#    Ör: "Köpekler ne yer?" + "Köpekler mama yer" → 0.95
#         "Köpekler ne yer?" + "Arabalar benzinle çalışır" → 0.05
# ══════════════════════════════════════════════════════════════════════


class Reranker(ABC):
    """
    Reranker arayüzü.
    Tüm reranker'lar rerank() metodunu uygulamalıdır.
    """
    @abstractmethod
    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]: ...


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder ile reranking.
    sentence-transformers kütüphanesindeki CrossEncoder modelini kullanır.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            logger.info(f"Cross-encoder yüklendi: {model_name}")
        except ImportError:
            logger.error("sentence-transformers gerekli: pip install sentence-transformers")
            raise

    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """
        Sorgu ve her belgeyi çift olarak modele ver, puan al.
        Puana göre sırala, en yüksek top_k belgeyi döndür.
        """
        # [sorgu, belge] çiftlerinden oluşan liste
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)  # Her çift için 0-1 arası puan
        scored = list(zip(scores, documents))  # (puan, belge) çiftleri
        scored.sort(key=lambda x: x[0], reverse=True)  # Puana göre azalan sırala
        return [doc for _, doc in scored[:top_k]]  # En iyi top_k belge


class RerankerFactory:
    """
    Reranker fabrikası.
    Config'de use_reranker=False ise None döndürür (reranker kullanılmaz).
    """
    @staticmethod
    def create(config: RAGConfig) -> Optional[Reranker]:
        if not config.use_reranker:
            return None
        return CrossEncoderReranker(model_name=config.reranker_model)


# ══════════════════════════════════════════════════════════════════════
#  8. SORGU DÖNÜŞÜMÜ (Query Transformation)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: Kullanıcının sorusunu daha etkili arama yapılabilecek hale getirmek.
#
#  HyDE (Hypothetical Document Embedding):
#    1. LLM'den soruya hayali bir cevap yazmasını iste
#    2. Orijinal soru + hayali cevap ile vektör araması yap
#    → Soru kısa ve özse bile, cevapla arama daha iyi eşleşme sağlar
#
#  Multi-Query:
#    1. LLM'den sorunun farklı versiyonlarını üretmesini iste
#    2. Tüm versiyonlarla ayrı ayrı vektör araması yap
#    3. Sonuçları birleştir ve benzersiz olanları al
#    → Aynı soruya farklı açılardan yaklaşarak daha kapsamlı sonuç
# ══════════════════════════════════════════════════════════════════════


class QueryTransformer(ABC):
    """Sorgu dönüştürücü arayüzü."""
    @abstractmethod
    def transform(self, query: str) -> Union[str, List[str]]: ...
    # str dönerse → tek sorgu (HyDE)
    # List[str] dönerse → çoklu sorgu (Multi-Query)


class HyDEQueryTransformer(QueryTransformer):
    """
    HyDE: Hypothetical Document Embedding.
    Önce LLM'den soruya hayali bir cevap üretmesini iste.
    Sonra soru + cevap birleştirilmiş halini aramada kullan.
    """
    def __init__(self, llm: BaseLanguageModel):
        # LCEL zinciri: prompt → LLM → string çıktı
        self.chain = (
            PromptTemplate.from_template(
                "Soruya kısa bir cevap yaz: {question}"
            )
            | llm
            | StrOutputParser()
        )

    def transform(self, query: str) -> Union[str, List[str]]:
        hypothetical_answer = self.chain.invoke({"question": query})
        combined = f"{query}\nCevap: {hypothetical_answer}"
        logger.info(f"HyDE dönüşümü: {query} → {combined[:100]}...")
        return combined


class MultiQueryTransformer(QueryTransformer):
    """
    Multi-Query: Sorunun farklı versiyonlarını üret, hepsiyle ara.
    """
    def __init__(self, llm: BaseLanguageModel, num_queries: int = 3):
        self.num_queries = num_queries
        self.chain = (
            PromptTemplate.from_template(
                """Aşağıdaki sorunun {num_queries} farklı versiyonunu üret.
Her versiyon farklı bir açıdan soruyu ele alsın.
Her satıra bir soru yaz, başka bir şey yazma.

Orijinal soru: {question}"""
            )
            | llm
            | StrOutputParser()
        )

    def transform(self, query: str) -> Union[str, List[str]]:
        result = self.chain.invoke({"question": query, "num_queries": self.num_queries})
        # LLM çıktısını satır satır ayır, boş satırları at
        queries = [q.strip() for q in result.split("\n") if q.strip()]
        queries.insert(0, query)  # Orijinal soruyu da başa ekle
        logger.info(f"Multi-Query: {len(queries)} sorgu üretildi")
        return queries


class QueryTransformerFactory:
    """Sorgu dönüştürücü fabrikası."""
    @staticmethod
    def create(config: RAGConfig, llm: BaseLanguageModel) -> Optional[QueryTransformer]:
        if config.use_hyde:
            return HyDEQueryTransformer(llm)
        elif config.use_multi_query:
            return MultiQueryTransformer(llm, num_queries=config.multi_query_count)
        return None


# ══════════════════════════════════════════════════════════════════════
#  9. KONUŞMA HAFIZASI (Conversation Memory)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: Çok turlu sohbetlerde önceki mesajları hatırlamak.
#
#  Neden gerekli?
#    - Kullanıcı "Peki ya ikinci madde?" diye sorabilir
#    - LLM'in "ikinci madde"den ne kastettiğini bilmesi gerekir
#    - Hafıza olmazsa her soru bağımsız işlenir, bağlam kaybolur
#
#  BufferMemory: Tüm mesajları saklar (büyüyebilir)
#  WindowMemory: Sadece son N mesajı saklar (sabit boyutlu)
# ══════════════════════════════════════════════════════════════════════


class ConversationMemory(ABC):
    """Konuşma hafızası arayüzü."""
    @abstractmethod
    def add_user_message(self, message: str): ...
    @abstractmethod
    def add_ai_message(self, message: str): ...
    @abstractmethod
    def get_history(self) -> List: ...
    @abstractmethod
    def clear(self): ...


class BufferMemory(ConversationMemory):
    """
    Tüm mesajları saklayan hafıza.
    Basit: Her mesajı bir listeye ekler, hepsini döndürür.
    Dezavantaj: Uzun sohbetlerde çok büyüyebilir.
    """
    def __init__(self):
        self.history: List = []  # Mesaj listesi: [HumanMessage, AIMessage, ...]

    def add_user_message(self, message: str):
        self.history.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.history.append(AIMessage(content=message))

    def get_history(self) -> List:
        return self.history  # Tüm geçmiş

    def clear(self):
        self.history = []


class WindowMemory(ConversationMemory):
    """
    Sadece son N mesajı saklayan hafıza.
    Avantaj: Hafıza boyutu sabit, sonsuz büyümez.
    Dezavantaj: Çok eski mesajlar unutulur.
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: List = []

    def add_user_message(self, message: str):
        self.history.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.history.append(AIMessage(content=message))

    def get_history(self) -> List:
        # Sadece son window_size*2 mesaj (user+ai çiftleri)
        return self.history[-self.window_size * 2:]

    def clear(self):
        self.history = []


class MemoryFactory:
    """Hafıza fabrikası."""
    @staticmethod
    def create(config: RAGConfig) -> Optional[ConversationMemory]:
        if not config.use_memory:
            return None  # Hafıza kullanılmayacak
        if config.memory_type == "window":
            return WindowMemory(window_size=config.memory_window)
        return BufferMemory()


# ══════════════════════════════════════════════════════════════════════
#  10. LLM SAĞLAYICI (LLM Factory)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: Seçilen LLM sağlayıcısına göre doğru sınıfı oluşturmak.
#  OpenAI: ChatOpenAI (GPT-4, GPT-3.5)
#  Ollama: ChatOllama (Llama 3, Mistral, Gemma vb.)
# ══════════════════════════════════════════════════════════════════════


class LLMFactory:
    """
    LLM fabrikası: config'de belirtilen sağlayıcıya göre
    uygun LLM nesnesini oluşturur.
    """
    @staticmethod
    def create(config: RAGConfig) -> BaseLanguageModel:
        if config.llm_provider == LLMProvider.OPENAI:
            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY gerekli!")
            # ChatOpenAI: OpenAI chat completion API'sine bağlanır
            return ChatOpenAI(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                openai_api_key=config.openai_api_key,
            )
        elif config.llm_provider == LLMProvider.OLLAMA:
            # ChatOllama: Yerel Ollama sunucusuna bağlanır
            return ChatOllama(
                model=config.llm_model,
                temperature=config.llm_temperature,
                base_url=config.ollama_base_url,
            )
        elif config.llm_provider == LLMProvider.HUGGINGFACE:
            # HuggingFace: Transformers ile yerel model yükler
            return HuggingFaceLLMFactory._create(config)
        raise ValueError(f"Bilinmeyen LLM provider: {config.llm_provider}")


class HuggingFaceLLMFactory:
    """
    HuggingFace modellerini yerel olarak yükleyen fabrika.
    transformers + bitsandbytes ile quantization desteği sunar.
    6GB VRAM için 4-bit quantization önerilir.
    """

    _cache: dict = {}  # Model önbelleği (aynı model tekrar yüklenmez)

    @staticmethod
    def _create(config: RAGConfig) -> BaseLanguageModel:
        from langchain_huggingface import HuggingFacePipeline
        import torch

        # Önbellekte varsa direkt döndür
        cache_key = f"{config.hf_llm_model}_{config.hf_quantize}"
        if cache_key in HuggingFaceLLMFactory._cache:
            logger.info(f"Model önbellekten yüklendi: {config.hf_llm_model}")
            return HuggingFaceLLMFactory._cache[cache_key]

        logger.info(f"HuggingFace modeli yükleniyor: {config.hf_llm_model}")

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline as hf_pipeline,
        )

        # Quantization config (VRAM tasarrufu)
        quantization_config = None
        if config.hf_quantize:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("4-bit quantization aktif (VRAM tasarrufu)")
            except Exception as e:
                logger.warning(f"bitsandbytes yok, quantization atlanıyor: {e}")
                quantization_config = None

        # Tokenizer'ı yükle
        tokenizer = AutoTokenizer.from_pretrained(
            config.hf_llm_model,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Modeli yükle
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_llm_model,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if not config.hf_quantize else None,
            device_map=config.hf_device,
            trust_remote_code=True,
        )

        # Pipeline oluştur
        pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.hf_max_new_tokens,
            temperature=config.llm_temperature,
            do_sample=config.llm_temperature > 0,
            return_full_text=False,  # Sadece yeni üretilen metni döndür
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # HuggingFacePipeline'e sar
        base_llm = HuggingFacePipeline(pipeline=pipe)

        # ChatHuggingFace: chat template'i otomatik uygular
        # (SmolLM2, Gemma gibi modeller için gerekli)
        from langchain_huggingface import ChatHuggingFace
        # pipeline.tokenizer üzerinden chat template alınır
        chat_llm = ChatHuggingFace(llm=base_llm)

        # Önbelleğe kaydet
        HuggingFaceLLMFactory._cache[cache_key] = chat_llm
        logger.info(f"Model hazir: {config.hf_llm_model}")
        return chat_llm


# ══════════════════════════════════════════════════════════════════════
#  11. ANA RAG SINIFI (RAGSystem)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: Tüm RAG bileşenlerini birleştiren ana sınıf.
#
#  RAG Akışı (A'dan Z'ye):
#    Yükle (load) → Böl (chunk) → Vektörleştir (embed)
#    → Depola (store) → Ara (retrieve) → Zenginleştir (rerank/transform)
#    → Üret (generate)
#
#  Bu sınıf:
#    - Config'i alır
#    - Tüm alt bileşenleri oluşturur
#    - Index yönetimini yapar (oluştur/kaydet/yükle)
#    - Sorguları yanıtlar
#    - Streaming desteği sunar
#    - Sistem bilgisini döndürür
# ══════════════════════════════════════════════════════════════════════


class RAGSystem:
    """
    Tüm RAG bileşenlerini birleştiren ana sınıf.
    A'dan Z'ye: yükle → böl → embed → depola → ara → zenginleştir → üret.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        RAG sistemini başlatır.
        1. Config'i al (yoksa varsayılan)
        2. Config'i doğrula
        3. Tüm bileşenleri oluştur
        """
        self.config = config or RAGConfig()
        self._validate_config()

        # Ana bileşenler (başlangıçta hemen oluşturulur)
        self.embeddings = EmbeddingFactory.create(self.config)
        self.llm = LLMFactory.create(self.config)

        # Alt bileşenler (index oluşturulana/yüklenene kadar None)
        self.vectorstore: Optional[VectorStore] = None       # Vektör veritabanı
        self.retriever: Optional[VectorStoreRetriever] = None # Belge getirici
        self.reranker: Optional[Reranker] = None              # Yeniden sıralayıcı
        self.query_transformer: Optional[QueryTransformer] = None  # Sorgu dönüştürücü
        self.memory: Optional[ConversationMemory] = None      # Konuşma hafızası
        self.chain: Optional[Runnable] = None                 # RAG zinciri

        # Belge ve parça listeleri
        self.documents: List[Document] = []  # Yüklenen orijinal belgeler
        self.chunks: List[Document] = []     # Bölünmüş parçalar

        # İsteğe bağlı bileşenleri oluştur (reranker, transformer, memory)
        self._init_optional_components()

        logger.info(
            f"RAG sistemi hazır | LLM: {self.config.llm_model} | "
            f"Embedding: {self.config.embedding_model} | "
            f"Vektör: {self.config.vector_store.value}"
        )

    def _validate_config(self):
        """Config'in geçerli olduğunu doğrula."""
        if self.config.llm_provider == LLMProvider.OPENAI and not self.config.openai_api_key:
            raise ValueError("OpenAI kullanımı için OPENAI_API_KEY gerekli!")
        if self.config.vector_store == VectorStoreType.CHROMA:
            # Chroma için index klasörünü oluştur
            os.makedirs(self.config.index_directory, exist_ok=True)

    def _init_optional_components(self):
        """
        İsteğe bağlı bileşenleri oluştur:
        - Reranker (use_reranker=True ise)
        - Query Transformer (use_hyde/use_multi_query=True ise)
        - Memory (use_memory=True ise)
        """
        self.reranker = RerankerFactory.create(self.config)
        self.query_transformer = QueryTransformerFactory.create(self.config, self.llm)
        self.memory = MemoryFactory.create(self.config)

    # ── Index Yönetimi ───────────────────────────────────────────────

    def index_documents(self, source: str) -> int:
        """
        Belge(ler)i yükle, parçalara böl, vektörleştir ve depola.
        Bu metot RAG'ın ilk ve en önemli adımıdır.

        Akış:
          1. Belge(ler)i yükle (DocumentLoaderFactory)
          2. Parçalara böl (ChunkerFactory)
          3. Vektörleştir + depola (VectorStoreFactory.from_documents)
          4. Vektör deposunu diske kaydet
          5. Retriever oluştur
          6. RAG zincirini oluştur

        Args:
            source: Dosya yolu, klasör yolu veya URL

        Returns:
            Oluşturulan parça (chunk) sayısı
        """
        logger.info(f"📄 Index başlıyor: {source}")

        # 1. Yükle: kaynaktaki belgeleri Document listesine çevir
        self.documents = DocumentLoaderFactory.load(source)
        if not self.documents:
            raise ValueError("Hiç belge yüklenemedi!")

        # 2. Böl: belgeleri küçük parçalara ayır
        chunker = ChunkerFactory.create(
            self.config.chunk_strategy,
            self.config.chunk_size,
            self.config.chunk_overlap,
        )
        self.chunks = chunker.split(self.documents)
        logger.info(f"✂️  {len(self.chunks)} parça oluşturuldu")

        # 3. Embed et ve depola: parçaları vektöre çevir, veritabanına ekle
        self.vectorstore = VectorStoreFactory.from_documents(
            self.config, self.chunks, self.embeddings
        )

        # 4. Diske kaydet (bir daha ki sefere hızlı yüklemek için)
        self._persist_vectorstore()

        # 5. Retriever'ı oluştur (aramaları yapacak nesne)
        self._build_retriever()

        # 6. RAG zincirini oluştur (soru → arama → yanıt)
        self._build_chain()

        logger.info(f"✅ Index tamam: {len(self.chunks)} parça, {len(self.documents)} belge")
        return len(self.chunks)

    def load_index(self) -> bool:
        """
        Daha önce kaydedilmiş index'i diskten yükler.
        Her seferinde belgeleri yeniden işlemekten kaçınır.

        Returns:
            Başarılıysa True, index bulunamazsa False
        """
        index_path = Path(self.config.index_directory)
        if not index_path.exists():
            logger.warning(f"Index bulunamadı: {self.config.index_directory}")
            return False

        try:
            # Diskteki index'i yükle
            self.vectorstore = VectorStoreFactory.load_local(self.config, self.embeddings)
            self._build_retriever()
            self._build_chain()
            logger.info(f"✅ Index yüklendi: {self.config.index_directory}")
            return True
        except Exception as e:
            logger.error(f"Index yüklenemedi: {e}")
            return False

    def _build_retriever(self):
        """Vektör deposundan bir retriever (belge getirici) oluşturur."""
        if self.vectorstore:
            self.retriever = RetrieverFactory.create(self.vectorstore, self.config)

    def _persist_vectorstore(self):
        """Vektör deposunu diske kaydeder (FAISS için)."""
        if self.config.vector_store == VectorStoreType.FAISS and self.vectorstore:
            os.makedirs(self.config.index_directory, exist_ok=True)
            self.vectorstore.save_local(self.config.index_directory)
            logger.info(f"💾 FAISS kaydedildi: {self.config.index_directory}")
        elif self.config.vector_store == VectorStoreType.CHROMA and self.vectorstore:
            pass  # Chroma zaten her eklemede otomatik kaydeder

    def _build_chain(self):
        """
        RAG zincirini oluşturur.
        LCEL (LangChain Expression Language) kullanır.
        | → pipe operatörü: bir adımın çıktısı diğerine giriş olur.

        Zincir akışı:
          soru → retriever (vektör araması) → prompt (soru+bağlam) → LLM → yanıt
        """
        if not self.vectorstore or not self.retriever:
            raise RuntimeError("Önce index oluştur veya yükle!")

        # PromptTemplate: şablondaki {context} ve {question} yerine
        # gerçek değerleri koyar
        prompt = ChatPromptTemplate.from_template(self.config.system_prompt)

        def format_docs(docs):
            """
            Document listesini okunabilir metne çevirir.
            Her belgeye [1], [2] gibi numara verir.
            """
            return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

        def retrieve_with_rerank(query: str) -> List[Document]:
            """
            Belge getir + isteğe bağlı reranker uygula.
            Reranker varsa: sonuçları daha hassas sırala.
            """
            docs = self.retriever.invoke(query)
            if self.reranker:
                docs = self.reranker.rerank(query, docs, self.config.reranker_top_k)
            return docs

        def process_query(query: str) -> Dict[str, Any]:
            """
            Sorgu dönüşümü varsa uygula.
            - HyDE: sorgu + hayali cevap ile ara
            - Multi-Query: her versiyonla ayrı ayrı ara, birleştir
            """
            if self.query_transformer:
                transformed = self.query_transformer.transform(query)
                if isinstance(transformed, list):
                    # Multi-Query: her sorguyla ayrı ayrı ara
                    all_docs = []
                    for q in transformed:
                        all_docs.extend(retrieve_with_rerank(q))
                    # Tekrar eden belgeleri çıkar, ilk k kadarını al
                    docs = self._deduplicate_documents(all_docs)[:self.config.retrieval_k]
                else:
                    # HyDE: dönüştürülmüş sorguyla ara
                    docs = retrieve_with_rerank(str(transformed))
            else:
                docs = retrieve_with_rerank(query)

            # Konuşma hafızası varsa geçmişi ekle
            history = self.memory.get_history() if self.memory else []

            return {
                "context": format_docs(docs),
                "question": query,
                "chat_history": history,
                "source_documents": docs,
            }

        # RAG zincirini oluştur (LCEL)
        # RunnablePassthrough: önce context'i hesapla, sonra prompt'a ekle
        self.chain = RunnablePassthrough.assign(
            context=lambda x: format_docs(retrieve_with_rerank(x["question"]))
            if not self.query_transformer
            else process_query(x["question"])["context"]
        )

    def _deduplicate_documents(self, docs: List[Document]) -> List[Document]:
        """
        Multi-Query sonrası tekrar eden belgeleri temizler.
        İlk 100 karaktere göre benzersizlik kontrolü yapar.
        """
        seen = set()
        unique = []
        for doc in docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    # ── Sorgulama Metodları ──────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        """
        Tek bir soruya RAG ile yanıt üretir.

        Akış:
          1. Sorgu dönüşümü varsa uygula (HyDE/Multi-Query)
          2. Vektör deposunda ara (retriever)
          3. Reranker varsa sonuçları yeniden sırala
          4. Prompt'u hazırla (sistem prompt + bağlam + soru)
          5. LLM'e gönder ve yanıt al
          6. Hafızaya kaydet (varsa)
          7. Yanıt + kaynakları döndür

        Args:
            question: Kullanıcının doğal dilde sorusu

        Returns:
            {
                "answer": LLM yanıtı (string),
                "source_documents": kullanılan belgeler (List[Document]),
                "context": prompt'a eklenen bağlam metni (string)
            }
        """
        if not self.chain:
            raise RuntimeError("Zincir hazır değil! Önce index_documents() veya load_index() çağırın.")

        # 1. Sorgu dönüşümü
        if self.query_transformer:
            transformed = self.query_transformer.transform(question)
            if isinstance(transformed, list):
                all_docs = []
                for q in transformed:
                    all_docs.extend(self.retriever.invoke(q) if self.retriever else [])
                docs = self._deduplicate_documents(all_docs)[:self.config.retrieval_k]
            else:
                docs = self.retriever.invoke(str(transformed)) if self.retriever else []
        else:
            docs = self.retriever.invoke(question) if self.retriever else []

        # 2. Reranker
        if self.reranker and docs:
            docs = self.reranker.rerank(question, docs, self.config.reranker_top_k)

        # 3. Prompt'u hazırla (chat template ile formatlanır)
        context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
        # Sistem promptundaki talimat kısmını al ({context} öncesi, "Bağlam:" satırını atla)
        system_text = self.config.system_prompt.split("\nBağlam:")[0].strip()
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=f"Bağlam:\n{context}\n\nSoru: {question}"),
        ]

        # 4. LLM çağrısı (ChatHuggingFace mesaj formatını otomatik işler)
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        # 5. Hafızaya kaydet
        if self.memory:
            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)

        return {
            "answer": answer,
            "source_documents": docs,
            "context": context,
        }

    def stream_query(self, question: str) -> Generator[str, None, None]:
        """
        Yanıtı streaming (akış) olarak üretir.
        Token token gelir → kullanıcı yanıtı anlık görür.

        Args:
            question: Kullanıcı sorusu

        Yields:
            Her token bir string olarak
        """
        if not self.chain:
            raise RuntimeError("Zincir hazır değil!")

        docs = self.retriever.invoke(question) if self.retriever else []
        if self.reranker and docs:
            docs = self.reranker.rerank(question, docs, self.config.reranker_top_k)

        context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
        system_text = self.config.system_prompt.split("\nBağlam:")[0].strip()
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=f"Bağlam:\n{context}\n\nSoru: {question}"),
        ]

        # LLM'in stream metodunu kullan: her token ayrı ayrı gelir
        for chunk in self.llm.stream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield content  # Generator: her yield'de bir token

        if self.memory:
            self.memory.add_user_message(question)
            self.memory.add_ai_message(context)

    def query_with_history(self, question: str) -> Dict[str, Any]:
        """
        Konuşma geçmişini de dikkate alarak soru sorar.
        "Peki ya ikinci madde?" gibi takip sorularını anlar.

        Args:
            question: Kullanıcının yeni sorusu

        Returns:
            query() ile aynı formatta yanıt
        """
        if not self.memory:
            return self.query(question)

        # Geçmiş mesajları al
        history = self.memory.get_history()
        if history:
            # Geçmişi metne çevir
            history_text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in history[-self.config.memory_window * 2:]
            )
            # Soruyu geçmişle birleştir → bağlamlı sorgu
            contextualized_question = f"Önceki konuşma:\n{history_text}\n\nYeni soru: {question}"
        else:
            contextualized_question = question

        return self.query(contextualized_question)

    # ── Sistem Bilgisi ───────────────────────────────────────────────

    def get_info(self) -> Dict[str, Any]:
        """Sistem yapılandırması ve istatistikleri döndürür."""
        return {
            "config": {
                "llm": f"{self.config.llm_provider.value}/{self.config.llm_model}",
                "embedding": f"{self.config.embedding_provider.value}/{self.config.embedding_model}",
                "vector_store": self.config.vector_store.value,
                "chunk_strategy": self.config.chunk_strategy.value,
                "retrieval_strategy": self.config.retrieval_strategy.value,
                "use_reranker": self.config.use_reranker,
                "use_hyde": self.config.use_hyde,
                "use_multi_query": self.config.use_multi_query,
                "use_memory": self.config.use_memory,
            },
            "stats": {
                "documents": len(self.documents),
                "chunks": len(self.chunks),
            },
        }


# ══════════════════════════════════════════════════════════════════════
#  12. DEĞERLENDİRME (Evaluation)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: RAG sisteminin ne kadar iyi çalıştığını ölçmek.
#
#  3 metrik:
#    Faithfulness (sadakat): Cevap, bağlamdaki bilgilerle çelişiyor mu?
#    Relevancy (alakalılık): Cevap, soruyla alakalı mı?
#    Accuracy (doğruluk):   Cevap, referans cevapla uyumlu mu? (ground truth varsa)
#
#  Her metrik: LLM (jüri) tarafından 0 veya 1 olarak puanlanır.
# ══════════════════════════════════════════════════════════════════════


class RAGEvaluator:
    """
    RAG sistemi değerlendiricisi.
    Bir LLM'i jüri olarak kullanır: her soru-cevap çiftini puanlar.
    """
    def __init__(self, config: RAGConfig):
        self.config = config
        # Jüri LLM: düşük temperature (tutarlı değerlendirme için)
        self.eval_llm = ChatOpenAI(
            model=config.eval_llm_model,
            temperature=0,  # 0 = tamamen deterministik
            openai_api_key=config.openai_api_key,
        )

    def evaluate(
        self,
        rag: RAGSystem,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Bir dizi soruyu RAG sistemine sorar ve metrikleri hesaplar.

        Args:
            rag: Değerlendirilecek RAG sistemi
            questions: Soru listesi
            ground_truths: Referans cevaplar (varsa accuracy hesaplanır)

        Returns:
            {faithfulness, relevancy, accuracy, samples}
        """
        results = {"total": 0, "faithfulness": 0, "relevancy": 0, "accuracy": 0}

        for i, question in enumerate(questions):
            # RAG'e sor
            result = rag.query(question)
            answer = result["answer"]
            context = result["context"]

            # Metrikleri hesapla (her biri 0 veya 1)
            faithfulness = self._score_faithfulness(answer, context)
            relevancy = self._score_relevancy(answer, question)

            results["faithfulness"] += faithfulness
            results["relevancy"] += relevancy

            # Ground truth varsa accuracy hesapla
            if ground_truths and i < len(ground_truths):
                accuracy = self._score_accuracy(answer, ground_truths[i])
                results["accuracy"] += accuracy

            results["total"] += 1

        # Ortalamaları hesapla
        n = results["total"]
        return {
            "faithfulness": round(results["faithfulness"] / n, 4) if n else 0,
            "relevancy": round(results["relevancy"] / n, 4) if n else 0,
            "accuracy": round(results["accuracy"] / n, 4) if ground_truths and n else 0,
            "samples": n,
        }

    def _score_faithfulness(self, answer: str, context: str) -> float:
        """
        Faithfulness (sadakat): Cevap, bağlamdaki bilgilerle çelişiyor mu?
        LLM'e sor: bağlam ve cevabı karşılaştır, çelişki var mı?
        """
        prompt = f"""Aşağıdaki bağlam ve cevabı karşılaştır.
Cevap bağlamdaki bilgilerle çelişiyor mu?
Sadece 0 (çelişiyor) veya 1 (çelişmiyor) yaz.

Bağlam:
{context[:1500]}

Cevap:
{answer}

Puan (0/1):"""
        try:
            resp = self.eval_llm.invoke(prompt)
            return 1 if "1" in resp.content else 0
        except:
            return 0

    def _score_relevancy(self, answer: str, question: str) -> float:
        """
        Relevancy (alakalılık): Cevap, soruyla alakalı mı?
        """
        prompt = f"""Cevap soruyla alakalı mı?
Sadece 0 (alakasız) veya 1 (alakalı) yaz.

Soru: {question}
Cevap: {answer}

Puan (0/1):"""
        try:
            resp = self.eval_llm.invoke(prompt)
            return 1 if "1" in resp.content else 0
        except:
            return 0

    def _score_accuracy(self, answer: str, ground_truth: str) -> float:
        """
        Accuracy (doğruluk): Cevap, referans cevapla uyumlu mu?
        """
        prompt = f"""Cevap, referans cevapla ne kadar uyumlu?
Sadece 0 (uyumsuz) veya 1 (uyumlu) yaz.

Referans: {ground_truth}
Cevap: {answer}

Puan (0/1):"""
        try:
            resp = self.eval_llm.invoke(prompt)
            return 1 if "1" in resp.content else 0
        except:
            return 0


# ══════════════════════════════════════════════════════════════════════
#  13. FASTAPI SERVİSİ (REST API)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: RAG sistemini HTTP API olarak sunmak.
#  Diğer uygulamalar (web, mobil) bu API üzerinden sorgu gönderebilir.
#
#  Endpoint'ler:
#    GET  /health  → Sağlık kontrolü
#    POST /query   → Soru sor
#    POST /stream/{question} → Streaming yanıt
#    POST /index   → Dosya yükle + index oluştur
#    GET  /info    → Sistem bilgisi
#
#  Kullanım:
#    python dosya.py api --port 8000
#    curl -X POST "http://localhost:8000/query" -H "..." -d '{"question": "..."}'
# ══════════════════════════════════════════════════════════════════════


class RAGAPI:
    """
    FastAPI ile RAG servisi.
    REST API üzerinden sorgulama, indexleme ve streaming yapmayı sağlar.
    """
    def __init__(self, rag: RAGSystem, host: str = "0.0.0.0", port: int = 8000):
        self.rag = rag
        self.host = host
        self.port = port
        self.app = self._create_app()

    def _create_app(self):
        """
        FastAPI uygulamasını oluşturur.
        Tüm endpoint'leri ve veri modellerini tanımlar.
        """
        from fastapi import FastAPI, HTTPException, UploadFile, File
        from pydantic import BaseModel

        # FastAPI uygulaması
        app = FastAPI(
            title="RAG API",
            version="1.0.0",
            description="A'dan Z'ye RAG Sistemi - API Dokümantasyonu",
        )

        # ── Veri Modelleri (Pydantic) ────────────────────────────
        # Gelen ve giden verinin şemasını tanımlar
        # Otomatik doğrulama + dokümantasyon

        class QueryRequest(BaseModel):
            """Sorgu isteği: soru + isteğe bağlı hafıza kullanımı."""
            question: str
            use_history: bool = False  # Konuşma geçmişi kullanılsın mı?

        class QueryResponse(BaseModel):
            """Sorgu yanıtı: cevap + kaynak belgeler."""
            answer: str
            sources: List[Dict[str, Any]]

        class IndexResponse(BaseModel):
            """Index oluşturma yanıtı."""
            status: str
            chunks: int
            documents: int

        # ── Endpoint'ler ─────────────────────────────────────────

        @app.get("/health")
        def health():
            """Sağlık kontrolü: sistem çalışıyor mu?"""
            return {"status": "ok", "service": "rag-api"}

        @app.post("/query", response_model=QueryResponse)
        def query(req: QueryRequest):
            """
            RAG sistemine soru sor.
            Request body: {"question": "Belgeler ne hakkında?", "use_history": false}
            """
            try:
                if req.use_history:
                    result = self.rag.query_with_history(req.question)
                else:
                    result = self.rag.query(req.question)

                # Kaynak belgeleri API yanıtına uygun formata çevir
                sources = [
                    {
                        "content": d.page_content[:300],  # İlk 300 karakter
                        "source": d.metadata.get("source", "unknown"),
                        "page": d.metadata.get("page", None),
                    }
                    for d in result.get("source_documents", [])
                ]
                return QueryResponse(answer=result["answer"], sources=sources)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/stream/{question}")
        def stream(question: str):
            """
            Streaming yanıt: token token veri akışı.
            SSE (Server-Sent Events) formatında döner.
            """
            from fastapi.responses import StreamingResponse

            def generate():
                # Her token ayrı ayrı üretilir ve istemciye gönderilir
                for token in self.rag.stream_query(question):
                    yield token

            return StreamingResponse(generate(), media_type="text/plain")

        @app.post("/index", response_model=IndexResponse)
        async def index_file(file: UploadFile = File(...)):
            """
            Bir dosyayı yükle, index oluştur.
            Desteklenen formatlar: PDF, TXT, DOCX, CSV, MD
            """
            try:
                suffix = Path(file.filename).suffix  # Dosya uzantısı
                # Geçici dosyaya kaydet
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = tmp.name

                # Index oluştur
                chunks = self.rag.index_documents(tmp_path)
                os.unlink(tmp_path)  # Geçici dosyayı sil

                return IndexResponse(
                    status="ok",
                    chunks=chunks,
                    documents=len(self.rag.documents),
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/info")
        def info():
            """Sistem yapılandırması ve istatistikler."""
            return self.rag.get_info()

        return app

    def run(self):
        """Uvicorn ile FastAPI sunucusunu başlat."""
        import uvicorn
        print(f"🌐 API başlatılıyor: http://localhost:{self.port}")
        print(f"📖 Doküman: http://localhost:{self.port}/docs")
        uvicorn.run(self.app, host=self.host, port=self.port)


# ══════════════════════════════════════════════════════════════════════
#  14. GRADIO ARAYÜZÜ (Web UI)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: RAG sistemini web arayüzü ile kullanmak.
#  Gradio: Python ile hızlı ML demo arayüzleri oluşturma kütüphanesi.
#
#  3 sekme:
#    Soru-Cevap:       Chatbot arayüzü (streaming yanıt)
#    Index Yönetimi:   Dosya yükle, index oluştur, sistem bilgisi
#    Ayarlar:          Mevcut yapılandırma bilgisi
#
#  Kullanım:
#    python dosya.py ui --port 7860
# ══════════════════════════════════════════════════════════════════════


class RAGUI:
    """
    Gradio ile web arayüzü.
    Kullanıcıların tarayıcı üzerinden RAG sistemini kullanmasını sağlar.
    """
    def __init__(self, rag: RAGSystem):
        self.rag = rag

    def launch(self, share: bool = False, port: int = 7860):
        """
        Gradio arayüzünü başlatır.

        Args:
            share: True ise geçici public URL oluşturur (ngrok)
            port:  Web arayüzünün çalışacağı port
        """
        import gradio as gr

        # ── Yardımcı fonksiyonlar ───────────────────────────────

        def answer_question(message, history):
            """Soruya yanıt üretir (non-streaming)."""
            result = self.rag.query(message)
            sources = "\n\n".join(
                f"📄 {d.metadata.get('source', '?')} (Sayfa {d.metadata.get('page', '-')})"
                for d in result["source_documents"][:3]
            )
            return f"{result['answer']}\n\n---\n📚 Kaynaklar:\n{sources}"

        def stream_answer(message, history):
            """
            Soruya streaming yanıt üretir.
            Her token geldikçe UI güncellenir.
            """
            full = ""
            for token in self.rag.stream_query(message):
                full += token
                yield full  # Gradio'ya kısmi yanıt

        def index_file(file):
            """Yüklenen dosyayı index'ler."""
            if file is None:
                return "Lütfen bir dosya seçin."
            try:
                chunks = self.rag.index_documents(file.name)
                return f"✅ Indexlendi: {chunks} parça, {len(self.rag.documents)} belge"
            except Exception as e:
                return f"❌ Hata: {e}"

        # ── Arayüz Tasarımı ─────────────────────────────────────

        with gr.Blocks(title="RAG Sistemi", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 📚 RAG Sistemi — A'dan Z'ye")

            # Sekme 1: Soru-Cevap
            with gr.Tab("💬 Soru-Cevap"):
                gr.ChatInterface(
                    fn=stream_answer,
                    title="RAG Chatbot",
                    description="Belgelerinizle ilgili sorular sorun. Yanıtlar anlık olarak gelir.",
                )

            # Sekme 2: Index Yönetimi
            with gr.Tab("📄 Index Yönetimi"):
                with gr.Row():
                    file_input = gr.File(
                        label="Belge Yükle",
                        file_types=[".pdf", ".txt", ".docx", ".csv", ".md"]
                    )
                    index_btn = gr.Button("Index Oluştur", variant="primary")
                index_output = gr.Textbox(label="Durum")

                index_btn.click(
                    fn=index_file,
                    inputs=[file_input],
                    outputs=[index_output],
                )

                gr.Markdown("---")
                gr.Markdown("### Mevcut Index")
                info_btn = gr.Button("Sistem Bilgisi")
                info_output = gr.JSON(label="Sistem Bilgisi")

                info_btn.click(
                    fn=self.rag.get_info,
                    inputs=[],
                    outputs=[info_output],
                )

            # Sekme 3: Ayarlar (salt okunur bilgi)
            with gr.Tab("⚙️  Ayarlar"):
                gr.Markdown(
                    "Ayarlar `RAGConfig` üzerinden yapılır.\n"
                    "Mevcut yapılandırma:\n"
                    f"- **LLM**: {self.rag.config.llm_model}\n"
                    f"- **Embedding**: {self.rag.config.embedding_model}\n"
                    f"- **Chunk size**: {self.rag.config.chunk_size}\n"
                    f"- **Retriever**: {self.rag.config.retrieval_strategy.value} (k={self.rag.config.retrieval_k})"
                )

        demo.queue()  # Sıra desteği (çoklu kullanıcı)
        demo.launch(share=share, server_port=port)


# ══════════════════════════════════════════════════════════════════════
#  15. CLI UYGULAMASI (Command Line Interface)
#  ──────────────────────────────────────────────────────────────────
#  Amaç: RAG sistemini komut satırından kullanmak.
#
#  Komutlar:
#    chat    → İnteraktif soru-cevap
#    index   → Belge index'leme
#    eval    → Değerlendirme
#    api     → FastAPI sunucusu başlat
#    ui      → Gradio arayüzü başlat
#    info    → Sistem bilgisi
#
#  Kullanım:
#    python dosya.py chat --source "./belgeler"
#    python dosya.py api --port 8000
#    python dosya.py eval --question "Ne anlatıyor?"
# ══════════════════════════════════════════════════════════════════════


class RAGCLI:
    """Komut satırı arayüzü (CLI)."""
    def __init__(self, rag: RAGSystem):
        self.rag = rag

    def run(self):
        """
        CLI'yi başlatır.
        Argümanları ayrıştırır ve uygun komutu çalıştırır.
        """
        import argparse

        # Argüman tanımları
        parser = argparse.ArgumentParser(
            description="RAG Sistemi CLI — A'dan Z'ye RAG",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Örnek kullanımlar:
  python rag_complete.py chat --source ./belgeler
  python rag_complete.py api --port 8000
  python rag_complete.py ui --port 7860
  python rag_complete.py eval --question "Belgeler ne hakkında?"
  python rag_complete.py info
            """
        )
        parser.add_argument(
            "command",
            nargs="?",
            default="chat",
            choices=["chat", "index", "eval", "api", "ui", "info"],
            help="Yapılacak işlem"
        )
        parser.add_argument("--source", "-s", help="Belge kaynağı (dosya/klasör yolu)")
        parser.add_argument("--question", "-q", help="Soru metni")
        parser.add_argument("--port", "-p", type=int, default=8000, help="API/UI port numarası")
        parser.add_argument("--share", action="store_true", help="Gradio arayüzünü herkese aç (public URL)")
        parser.add_argument("--eval-file", help="Değerlendirme için JSON dosyası")

        args = parser.parse_args()

        # ── info: Sistem bilgisi ────────────────────────────────
        if args.command == "info":
            info = self.rag.get_info()
            print(json.dumps(info, indent=2, ensure_ascii=False))

        # ── index: Belge index'le ───────────────────────────────
        elif args.command == "index":
            if not args.source:
                print("Hata: --source gerekli. Ör: --source ./belgeler")
                return
            chunks = self.rag.index_documents(args.source)
            print(f"✅ {chunks} parça indexlendi.")

        # ── chat: İnteraktif soru-cevap ─────────────────────────
        elif args.command == "chat":
            if self.rag.vectorstore is None:
                print("Henüz index yok. Önce 'index' komutunu kullanın.")
                return

            print("\n" + "=" * 60)
            print("  RAG CHAT — Çıkmak için 'q' yazın")
            print("  Önce belgeleri index'lemeyi unutmayın!")
            print("=" * 60)

            while True:
                question = input("\n❓ Soru: ").strip()
                if question.lower() in ("q", "quit", "exit", "çık"):
                    print("Görüşmek üzere!")
                    break
                if not question:
                    continue

                # RAG sorgusu
                result = self.rag.query(question)
                print(f"\n💬 Yanıt: {result['answer']}")

                # Kaynak belgeler
                print("📚 Kaynaklar:")
                for i, doc in enumerate(result["source_documents"][:3]):
                    src = doc.metadata.get("source", "?")
                    page = doc.metadata.get("page", "-")
                    print(f"   [{i+1}] {src} | Sayfa {page}")

        # ── eval: Değerlendirme ─────────────────────────────────
        elif args.command == "eval":
            if args.eval_file:
                with open(args.eval_file, encoding="utf-8") as f:
                    data = json.load(f)
                questions = data.get("questions", [])
                ground_truths = data.get("ground_truths", [])
            else:
                questions = [args.question or "Belgelerde ne anlatılıyor?"]
                ground_truths = []

            evaluator = RAGEvaluator(self.rag.config)
            results = evaluator.evaluate(self.rag, questions, ground_truths or None)
            print(json.dumps(results, indent=2, ensure_ascii=False))

        # ── api: FastAPI sunucusu ───────────────────────────────
        elif args.command == "api":
            api = RAGAPI(self.rag, port=args.port)
            api.run()

        # ── ui: Gradio arayüzü ──────────────────────────────────
        elif args.command == "ui":
            ui = RAGUI(self.rag)
            print(f"🎨 UI başlatılıyor: http://localhost:{args.port}")
            ui.launch(share=args.share, port=args.port)


# ══════════════════════════════════════════════════════════════════════
#  ANA PROGRAM (Entry Point)
#  ──────────────────────────────────────────────────────────────────
#  Bu blok sadece dosya doğrudan çalıştırıldığında çalışır.
#  (import edildiğinde __name__ != "__main__" olur, bu blok atlanır)
#
#  Akış:
#    1. Config dosyası varsa yükle, yoksa varsayılanı kullan
#    2. RAGSystem'i oluştur
#    3. Mevcut index varsa otomatik yükle
#    4. CLI'yi başlat
# ══════════════════════════════════════════════════════════════════════


def create_default_config() -> RAGConfig:
    """
    Varsayılan yapılandırmayı oluşturur.
    Değerleri önce çevresel değişkenlerden (env) okur,
    bulamazsa sabit varsayılanları kullanır.

    Çevresel değişkenler:
      OPENAI_API_KEY       → OpenAI API anahtarı
      RAG_LLM_MODEL        → LLM model adı
      RAG_EMBEDDING_MODEL  → Embedding model adı
      RAG_INDEX_DIR        → Index kayıt dizini
      RAG_CHUNK_SIZE       → Parça boyutu
      RAG_CHUNK_OVERLAP    → Parça örtüşmesi
      RAG_RETRIEVAL_K      → Getirilecek belge sayısı
      RAG_VERBOSE          → Ayrıntılı log

    YAML config dosyası (RAG_CONFIG env değişkeniyle belirtilir):
      Varsayılan: rag_config.yaml
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_openai = bool(api_key)

    if use_openai:
        provider = EmbeddingProvider.OPENAI
        llm_provider = LLMProvider.OPENAI
        llm_model = "gpt-4o-mini"
        emb_model = "text-embedding-ada-002"
    else:
        # HuggingFace modeli tercih ediliyor (Ollama'ya gerek yok)
        provider = EmbeddingProvider.HUGGINGFACE
        llm_provider = LLMProvider.HUGGINGFACE
        llm_model = os.getenv("RAG_LLM_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
        emb_model = "all-MiniLM-L6-v2"

    return RAGConfig(
        openai_api_key=api_key,
        llm_provider=llm_provider,
        embedding_provider=provider,
        llm_model=llm_model,
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", emb_model),
        hf_llm_model=os.getenv("RAG_HF_MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
        hf_quantize=os.getenv("RAG_HF_QUANTIZE", "true").lower() == "true",
        hf_max_new_tokens=int(os.getenv("RAG_HF_MAX_TOKENS", "512")),
        index_directory=os.getenv("RAG_INDEX_DIR", "./rag_index"),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
        retrieval_k=int(os.getenv("RAG_RETRIEVAL_K", "4")),
        verbose=os.getenv("RAG_VERBOSE", "false").lower() == "true",
    )


def main():
    """Ana giriş noktası — CLI'yi başlatır."""
    # Config dosya yolu (env değişkeninden veya varsayılan)
    config_path = os.getenv("RAG_CONFIG", "rag_config.yaml")

    # Config yükle: YAML varsa ondan, yoksa env/varsayılan
    if Path(config_path).exists():
        config = RAGConfig.from_yaml(config_path)
        logger.info(f"Config dosyasından yüklendi: {config_path}")
    else:
        config = create_default_config()
        logger.info("Varsayılan config kullanılıyor")

    # RAG sistemini başlat
    rag = RAGSystem(config)

    # Mevcut index varsa otomatik yükle
    if Path(config.index_directory).exists():
        rag.load_index()

    # CLI'yi başlat
    cli = RAGCLI(rag)
    cli.run()


# Python dosyası doğrudan çalıştırıldığında main()'i çağır
# Ör: python rag_complete.py chat --source ./belgeler
if __name__ == "__main__":
    main()
