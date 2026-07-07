"""
=============================================================================
ETL / ELT Pipeline Yönetimi — Apache Airflow Mimarisi Simülasyonu
=============================================================================

Bu dosya, gerçek bir Apache Airflow ortamında nasıl çalışacağını göstermek
için Airflow'un temel bileşenlerini (DAG, Task, Operator, XCom, TaskGroup)
Python ile sıfırdan simüle eder.

Gerçek Airflow ortamında:
  - Bu dosya $AIRFLOW_HOME/dags/ klasörüne kopyalanır.
  - `airflow dags trigger etl_satis_pipeline` komutuyla çalıştırılır.
  - Web UI üzerinden izlenir (localhost:8080).

Simülasyon sayesinde Airflow kurulmadan tüm mimari test edilebilir.

Kullanılan Airflow kavramları:
  ┌─────────────────────────────────────────────────────────────┐
  │  DAG          → Yönlendirilmiş Asiklik Graf (iş akışı)     │
  │  Task         → DAG içindeki tek bir iş birimi             │
  │  PythonOperator → Python fonksiyonu çalıştıran operatör    │
  │  XCom         → Task'lar arası veri paylaşım mekanizması   │
  │  TaskGroup    → Görevleri mantıksal gruplara ayırma         │
  │  Schedule     → Otomatik çalıştırma zamanlaması            │
  └─────────────────────────────────────────────────────────────┘

ETL Akışı:
  [EXTRACT] ──► [TRANSFORM] ──► [LOAD] ──► [VALIDATE] ──► [NOTIFY]
      │               │              │            │
   Ham veri       Temizlik      Hedef DB      Kalite         Rapor
   çekme          Dönüşüm       yazma         kontrolü      gönderme
=============================================================================
"""

import time
import json
import random
import hashlib
import sqlite3
import logging
import tempfile
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ── Bağımlılıklar (pip install pandas faker sqlalchemy) ──────────────────────
import pandas as pd
from faker import Faker

# ── Loglama Yapılandırması ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("airflow.simulation")

fake = Faker("tr_TR")  # Türkçe sahte veri üreteci


DEFAULT_DB_PATH = str(Path(tempfile.gettempdir()) / "satis_warehouse.db")


# =============================================================================
# ❶  AIRFLOW CORE — Temel Sınıflar (Simülasyon)
# =============================================================================

class TaskState(Enum):
    """Task'ın olası durumları (Airflow'da aynı enum mevcuttur)."""
    PENDING   = "pending"    # Henüz çalışmadı
    RUNNING   = "running"    # Şu an çalışıyor
    SUCCESS   = "success"    # Başarıyla tamamlandı
    FAILED    = "failed"     # Hata ile sonuçlandı
    SKIPPED   = "skipped"    # Atlandı


@dataclass
class XComStore:
    """
    XCom (Cross-Communication) — Task'lar arası mesaj kutusu.
    Gerçek Airflow'da bu veriler metadata DB'de saklanır.
    Küçük veri paylaşımı için tasarlanmıştır (büyük veri için S3/GCS kullanın).
    """
    _store: Dict[str, Any] = field(default_factory=dict)

    def push(self, task_id: str, key: str, value: Any):
        """Bir task'ın ürettiği veriyi depola."""
        store_key = f"{task_id}::{key}"
        self._store[store_key] = value
        log.debug(f"XCom PUSH → {store_key}")

    def pull(self, task_id: str, key: str) -> Any:
        """Bir task'ın daha önce depoladığı veriyi çek."""
        store_key = f"{task_id}::{key}"
        value = self._store.get(store_key)
        log.debug(f"XCom PULL ← {store_key} = {type(value).__name__}")
        return value


@dataclass
class TaskInstance:
    """
    Airflow TaskInstance: Bir task'ın tek bir çalışma kaydı.
    task_id, dag_id, execution_date bilgilerini tutar.
    """
    task_id: str
    dag_id: str
    execution_date: datetime
    xcom: XComStore
    state: TaskState = TaskState.PENDING
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration: float = 0.0
    log_messages: List[str] = field(default_factory=list)

    def xcom_push(self, key: str, value: Any):
        """Bu task'tan veri yayınla."""
        self.xcom.push(self.task_id, key, value)

    def xcom_pull(self, task_ids: str, key: str) -> Any:
        """Başka bir task'ın verisini çek."""
        return self.xcom.pull(task_ids, key)

    def log_info(self, msg: str):
        self.log_messages.append(msg)
        log.info(f"[{self.task_id}] {msg}")


class PythonOperator:
    """
    Airflow PythonOperator: Python fonksiyonu çalıştıran temel operatör.
    Gerçek kullanım:
        task = PythonOperator(
            task_id='my_task',
            python_callable=my_function,
            op_kwargs={'param': 'value'},
            dag=dag,
        )
    """
    def __init__(
        self,
        task_id: str,
        python_callable: Callable,
        op_kwargs: Optional[Dict] = None,
        retries: int = 1,
        retry_delay: int = 2,  # saniye
        group: str = "default",
    ):
        self.task_id = task_id
        self.python_callable = python_callable
        self.op_kwargs = op_kwargs or {}
        self.retries = retries
        self.retry_delay = retry_delay
        self.group = group
        self.upstream_tasks: List[str] = []    # Bu task'tan önce çalışacaklar
        self.downstream_tasks: List[str] = []  # Bu task'tan sonra çalışacaklar

    def set_upstream(self, task: "PythonOperator"):
        """Bağımlılık tanımla: self, task'tan SONRA çalışır."""
        if task.task_id not in self.upstream_tasks:
            self.upstream_tasks.append(task.task_id)
        if self.task_id not in task.downstream_tasks:
            task.downstream_tasks.append(self.task_id)

    def __rshift__(self, other: "PythonOperator"):
        """>> operatörü: task1 >> task2 şeklinde bağımlılık tanımlar."""
        other.set_upstream(self)
        return other


class DAG:
    """
    Airflow DAG (Directed Acyclic Graph): Tüm iş akışının konteyneri.
    
    Gerçek Airflow'da DAG dosyası $AIRFLOW_HOME/dags/ içine konur ve
    scheduler periyodik olarak tarar.
    
    Örnek schedule değerleri:
        "@daily"      → Her gün gece yarısı
        "@hourly"     → Her saat başı
        "0 6 * * *"   → Her gün 06:00'da (cron syntax)
        None          → Manuel tetikleme
    """
    def __init__(
        self,
        dag_id: str,
        description: str,
        schedule_interval: Optional[str],
        start_date: datetime,
        catchup: bool = False,
        tags: Optional[List[str]] = None,
        default_args: Optional[Dict] = None,
    ):
        self.dag_id = dag_id
        self.description = description
        self.schedule_interval = schedule_interval
        self.start_date = start_date
        self.catchup = catchup
        self.tags = tags or []
        self.default_args = default_args or {}
        self.tasks: Dict[str, PythonOperator] = {}

    def add_task(self, task: PythonOperator):
        """DAG'a görev ekle."""
        self.tasks[task.task_id] = task

    def run(self, execution_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        DAG'ı çalıştır — topolojik sıralama ile görevleri işle.
        Gerçek Airflow'da bu işi Executor (LocalExecutor, CeleryExecutor) yapar.
        """
        execution_date = execution_date or datetime.now()
        xcom = XComStore()

        print("\n" + "═" * 70)
        print(f"  🚀  DAG ÇALIŞIYOR: {self.dag_id}")
        print(f"  📅  Execution Date: {execution_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  📋  Açıklama: {self.description}")
        print("═" * 70)

        # Topolojik sıralama (bağımlılık sırasına göre çalıştır)
        ordered = self._topological_sort()
        results: Dict[str, TaskInstance] = {}
        dag_start = time.time()

        for task_id in ordered:
            task = self.tasks[task_id]
            ti = TaskInstance(
                task_id=task_id,
                dag_id=self.dag_id,
                execution_date=execution_date,
                xcom=xcom,
            )

            # Upstream task'ların başarılı olup olmadığını kontrol et
            failed_upstream = [
                uid for uid in task.upstream_tasks
                if results.get(uid) and results[uid].state == TaskState.FAILED
            ]
            if failed_upstream:
                ti.state = TaskState.SKIPPED
                ti.log_info(f"Atlandı — başarısız upstream: {failed_upstream}")
                results[task_id] = ti
                continue

            # Task'ı retry mantığıyla çalıştır
            self._execute_task(task, ti)
            results[task_id] = ti

        # Özet raporu
        dag_duration = time.time() - dag_start
        self._print_summary(results, dag_duration)
        return results

    def _execute_task(self, task: PythonOperator, ti: TaskInstance):
        """Tek bir task'ı yeniden deneme (retry) desteğiyle çalıştır."""
        group_icon = {"extract": "📥", "transform": "⚙️",
                      "load": "💾", "validate": "✅", "notify": "📣"
                      }.get(task.group, "🔧")

        print(f"\n  {group_icon}  Task: {ti.task_id}  [{task.group.upper()}]")
        print(f"      {'─' * 50}")

        for attempt in range(task.retries + 1):
            try:
                ti.state = TaskState.RUNNING
                ti.start_date = datetime.now()

                # Asıl fonksiyonu çalıştır — ti (TaskInstance) inject et
                task.python_callable(ti=ti, **task.op_kwargs)

                ti.end_date = datetime.now()
                ti.duration = (ti.end_date - ti.start_date).total_seconds()
                ti.state = TaskState.SUCCESS
                print(f"      ✓  Süre: {ti.duration:.2f}s")
                return

            except Exception as exc:
                if attempt < task.retries:
                    ti.log_info(f"Hata (deneme {attempt+1}/{task.retries}): {exc}")
                    print(f"      ⚠  Yeniden deneniyor ({attempt+1}/{task.retries})...")
                    time.sleep(task.retry_delay)
                else:
                    ti.end_date = datetime.now()
                    ti.duration = (ti.end_date - ti.start_date).total_seconds()
                    ti.state = TaskState.FAILED
                    ti.log_info(f"BAŞARISIZ: {exc}")
                    print(f"      ✗  HATA: {exc}")

    def _topological_sort(self) -> List[str]:
        """Kahn algoritması ile topolojik sıralama."""
        in_degree = {tid: 0 for tid in self.tasks}
        for task in self.tasks.values():
            for downstream in task.downstream_tasks:
                if downstream in in_degree:
                    in_degree[downstream] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        ordered = []

        while queue:
            tid = queue.pop(0)
            ordered.append(tid)
            for downstream in self.tasks[tid].downstream_tasks:
                if downstream in in_degree:
                    in_degree[downstream] -= 1
                    if in_degree[downstream] == 0:
                        queue.append(downstream)

        return ordered

    def _print_summary(self, results: Dict[str, TaskInstance], duration: float):
        """Pipeline sonuç özetini yazdır."""
        success = sum(1 for ti in results.values() if ti.state == TaskState.SUCCESS)
        failed  = sum(1 for ti in results.values() if ti.state == TaskState.FAILED)
        skipped = sum(1 for ti in results.values() if ti.state == TaskState.SKIPPED)

        print("\n" + "═" * 70)
        print(f"  📊  DAG TAMAMLANDI: {self.dag_id}")
        print(f"  ⏱   Toplam süre  : {duration:.2f} saniye")
        print(f"  ✅  Başarılı     : {success} task")
        print(f"  ❌  Başarısız    : {failed} task")
        print(f"  ⏭   Atlandı      : {skipped} task")
        print("═" * 70 + "\n")


# =============================================================================
# ❷  EXTRACT — Veri Çekme Görevleri
# =============================================================================

def extract_sales_data(ti: TaskInstance, num_records: int = 500):
    """
    [EXTRACT TASK] Kaynak sistemden (API / veritabanı) satış verisi çek.
    
    Gerçek senaryoda:
        - REST API'den JSON çekersiniz (requests.get)
        - PostgreSQL'den SELECT sorgusu çalıştırırsınız
        - CSV/Excel dosyasını okursunuz
        - Kafka topic'inden tüketirsiniz
    """
    ti.log_info(f"Kaynak sistemden {num_records} kayıt çekiliyor...")

    # Sahte satış verisi üret (gerçekte API/DB bağlantısı olur)
    records = []
    categories = ["Elektronik", "Giyim", "Gıda", "Mobilya", "Spor"]
    regions    = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]

    for _ in range(num_records):
        records.append({
            "order_id"    : fake.uuid4(),
            "customer_id" : f"CUST-{random.randint(1000, 9999)}",
            "product_name": fake.word().capitalize() + " " + fake.word(),
            "category"    : random.choice(categories),
            "region"      : random.choice(regions),
            "quantity"    : random.randint(1, 20),
            "unit_price"  : round(random.uniform(10.0, 5000.0), 2),
            "order_date"  : fake.date_between(
                start_date="-90d", end_date="today"
            ).isoformat(),
            # Kasıtlı kirli veri — TRANSFORM aşamasında temizlenecek
            "discount_pct": random.choice([0, 5, 10, 15, None, -5, 200]),
            "status"      : random.choice(
                ["completed", "COMPLETED", "pending", "cancelled", None]
            ),
        })

    df = pd.DataFrame(records)

    # XCom ile sonraki task'a ilet
    ti.xcom_push("raw_data", df.to_json(orient="records"))
    ti.xcom_push("record_count", len(df))
    ti.log_info(f"{len(df)} kayıt başarıyla çekildi.")


def extract_customer_data(ti: TaskInstance):
    """
    [EXTRACT TASK] Müşteri master data'sını çek.
    Gerçek senaryoda CRM veya MDM sisteminden okunur.
    """
    ti.log_info("Müşteri master verisi çekiliyor...")

    customers = [
        {
            "customer_id": f"CUST-{i}",
            "segment"    : random.choice(["Premium", "Standart", "Ekonomi"]),
            "join_date"  : fake.date_between("-3y", "-30d").isoformat(),
        }
        for i in range(1000, 10000)
    ]

    df = pd.DataFrame(customers)
    ti.xcom_push("customer_data", df.to_json(orient="records"))
    ti.log_info(f"{len(df)} müşteri kaydı çekildi.")


# =============================================================================
# ❸  TRANSFORM — Veri Dönüşüm Görevleri
# =============================================================================

def clean_and_validate_data(ti: TaskInstance):
    """
    [TRANSFORM TASK] Ham veriyi temizle ve standartlaştır.
    
    İşlemler:
        1. Null değerleri ele al
        2. Geçersiz aralıkları düzelt/filtrele
        3. Metin normalizasyonu (büyük/küçük harf)
        4. Veri tipi dönüşümleri
        5. Duplicate kayıtları kaldır
    """
    ti.log_info("Veri temizleme başlıyor...")

    # XCom'dan önceki task'ın verisini çek
    raw_json = ti.xcom_pull("extract_sales_data", "raw_data")
    df = pd.read_json(pd.io.common.StringIO(raw_json), orient="records")
    initial_count = len(df)

    # 1. Status normalizasyonu (büyük harf → küçük harf)
    df["status"] = df["status"].str.lower().str.strip()

    # 2. Geçersiz status değerlerini filtrele
    valid_statuses = {"completed", "pending", "cancelled"}
    df = df[df["status"].isin(valid_statuses)]

    # 3. İskonto değeri düzeltme (0-100 arası olmalı)
    df["discount_pct"] = df["discount_pct"].fillna(0)
    df["discount_pct"] = df["discount_pct"].clip(0, 100)

    # 4. Eksik kritik alanları kaldır
    df = df.dropna(subset=["order_id", "customer_id", "unit_price"])

    # 5. Duplicate order_id'leri kaldır (son kaydı tut)
    df = df.drop_duplicates(subset=["order_id"], keep="last")

    # 6. Tarih tipini düzelt
    df["order_date"] = pd.to_datetime(df["order_date"])

    removed = initial_count - len(df)
    ti.log_info(f"Temizleme tamamlandı: {removed} kayıt kaldırıldı, {len(df)} kaldı.")

    ti.xcom_push("clean_data", df.to_json(orient="records", date_format="iso"))
    ti.xcom_push("removed_count", removed)


def enrich_and_aggregate(ti: TaskInstance):
    """
    [TRANSFORM TASK] Veriyi zenginleştir ve iş metriklerini hesapla.
    
    İşlemler:
        1. Toplam tutar hesaplama
        2. Müşteri segmenti ile join
        3. Bölge bazlı özet (aggregation)
        4. Kategori bazlı KPI'lar
    """
    ti.log_info("Veri zenginleştirme ve agregasyon başlıyor...")

    # Temiz satış verisini çek
    clean_json    = ti.xcom_pull("clean_and_validate_data", "clean_data")
    customer_json = ti.xcom_pull("extract_customer_data", "customer_data")

    df_sales    = pd.read_json(pd.io.common.StringIO(clean_json), orient="records")
    df_customers = pd.read_json(pd.io.common.StringIO(customer_json), orient="records")

    # 1. Türetilmiş kolonlar
    df_sales["order_date"]    = pd.to_datetime(df_sales["order_date"])
    df_sales["gross_amount"]  = df_sales["quantity"] * df_sales["unit_price"]
    df_sales["discount_amount"] = df_sales["gross_amount"] * df_sales["discount_pct"] / 100
    df_sales["net_amount"]    = df_sales["gross_amount"] - df_sales["discount_amount"]
    df_sales["order_month"]   = df_sales["order_date"].dt.to_period("M").astype(str)
    df_sales["order_year"]    = df_sales["order_date"].dt.year

    # 2. Müşteri segmenti ile sol birleştirme (LEFT JOIN)
    df_enriched = df_sales.merge(
        df_customers[["customer_id", "segment"]],
        on="customer_id",
        how="left"
    )
    df_enriched["segment"] = df_enriched["segment"].fillna("Bilinmiyor")

    # 3. Bölge × Kategori bazlı özet
    region_summary = (
        df_enriched.groupby(["region", "category"])
        .agg(
            toplam_siparis=("order_id", "count"),
            toplam_ciro=("net_amount", "sum"),
            ort_sepet=("net_amount", "mean"),
            toplam_adet=("quantity", "sum"),
        )
        .round(2)
        .reset_index()
    )

    # 4. Aylık trend
    monthly_trend = (
        df_enriched.groupby("order_month")
        .agg(siparis=("order_id", "count"), ciro=("net_amount", "sum"))
        .round(2)
        .reset_index()
    )

    ti.xcom_push("enriched_data",   df_enriched.to_json(orient="records", date_format="iso"))
    ti.xcom_push("region_summary",  region_summary.to_json(orient="records"))
    ti.xcom_push("monthly_trend",   monthly_trend.to_json(orient="records"))
    ti.xcom_push("total_revenue",   round(df_enriched["net_amount"].sum(), 2))

    ti.log_info(
        f"Zenginleştirme tamamlandı. "
        f"Toplam ciro: ₺{df_enriched['net_amount'].sum():,.2f}"
    )


# =============================================================================
# ❹  LOAD — Hedef Sisteme Yazma Görevleri
# =============================================================================

def load_to_data_warehouse(ti: TaskInstance, db_path: str = DEFAULT_DB_PATH):
    """
    [LOAD TASK] Dönüştürülmüş veriyi Data Warehouse'a yaz.
    
    Gerçek senaryoda:
        - SQLAlchemy ile PostgreSQL/Redshift/Snowflake'e yazılır
        - df.to_sql("table", engine, if_exists="append") kullanılır
        - Büyük tablolar için COPY komutu veya bulk insert tercih edilir
    
    Burada SQLite ile gösterim yapılmaktadır.
    """
    ti.log_info(f"Data Warehouse'a yazılıyor: {db_path}")

    enriched_json = ti.xcom_pull("enrich_and_aggregate", "enriched_data")
    region_json   = ti.xcom_pull("enrich_and_aggregate", "region_summary")
    monthly_json  = ti.xcom_pull("enrich_and_aggregate", "monthly_trend")

    df_enriched      = pd.read_json(pd.io.common.StringIO(enriched_json), orient="records")
    df_region_summary = pd.read_json(pd.io.common.StringIO(region_json), orient="records")
    df_monthly       = pd.read_json(pd.io.common.StringIO(monthly_json), orient="records")

    # Hedef klasör yoksa oluştur (platform bağımsız).
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)

    try:
        # fact_sales → Ana satış gerçek tablosu
        df_enriched.to_sql("fact_sales", conn, if_exists="replace", index=False)
        ti.log_info(f"fact_sales: {len(df_enriched)} satır yazıldı.")

        # dim_region_category → Bölge × kategori özet boyutu
        df_region_summary.to_sql(
            "dim_region_category", conn, if_exists="replace", index=False
        )
        ti.log_info(f"dim_region_category: {len(df_region_summary)} satır yazıldı.")

        # fact_monthly_trend → Aylık trend gerçek tablosu
        df_monthly.to_sql("fact_monthly_trend", conn, if_exists="replace", index=False)
        ti.log_info(f"fact_monthly_trend: {len(df_monthly)} satır yazıldı.")

        # Yükleme meta verisi kaydet
        meta = pd.DataFrame([{
            "pipeline_run_id": hashlib.md5(
                str(ti.execution_date).encode()
            ).hexdigest()[:8],
            "loaded_at"      : datetime.now().isoformat(),
            "row_count"      : len(df_enriched),
        }])
        meta.to_sql("pipeline_runs", conn, if_exists="append", index=False)

        conn.commit()
        ti.xcom_push("db_path", db_path)
        ti.log_info("Tüm tablolar başarıyla yüklendi.")

    finally:
        conn.close()


# =============================================================================
# ❺  VALIDATE — Veri Kalite Kontrol Görevi
# =============================================================================

def run_data_quality_checks(ti: TaskInstance):
    """
    [VALIDATE TASK] Yüklenen veri üzerinde kalite kontrolleri çalıştır.
    
    Great Expectations veya Soda gibi araçlar bu aşamada kullanılır.
    Kontroller başarısız olursa pipeline başarısız sayılır ve
    downstream task'lar (bildirim hariç) atlanır.
    """
    ti.log_info("Veri kalite kontrolleri başlıyor...")

    db_path = ti.xcom_pull("load_to_data_warehouse", "db_path")
    conn    = sqlite3.connect(db_path)

    checks = []

    try:
        # ── Kontrol 1: Minimum kayıt sayısı ──────────────────────────────────
        count = pd.read_sql("SELECT COUNT(*) AS cnt FROM fact_sales", conn).iloc[0]["cnt"]
        checks.append({
            "name"   : "min_row_count",
            "passed" : count >= 100,
            "detail" : f"fact_sales kayıt sayısı: {count} (min: 100)",
        })

        # ── Kontrol 2: Negatif net_amount yok mu? ────────────────────────────
        neg_count = pd.read_sql(
            "SELECT COUNT(*) AS cnt FROM fact_sales WHERE net_amount < 0", conn
        ).iloc[0]["cnt"]
        checks.append({
            "name"   : "no_negative_amounts",
            "passed" : neg_count == 0,
            "detail" : f"Negatif tutar sayısı: {neg_count} (beklenen: 0)",
        })

        # ── Kontrol 3: Null order_id yok mu? ─────────────────────────────────
        null_ids = pd.read_sql(
            "SELECT COUNT(*) AS cnt FROM fact_sales WHERE order_id IS NULL", conn
        ).iloc[0]["cnt"]
        checks.append({
            "name"   : "no_null_order_ids",
            "passed" : null_ids == 0,
            "detail" : f"Null order_id sayısı: {null_ids} (beklenen: 0)",
        })

        # ── Kontrol 4: İskonto oranı mantıklı mı? ────────────────────────────
        bad_disc = pd.read_sql(
            "SELECT COUNT(*) AS cnt FROM fact_sales "
            "WHERE discount_pct < 0 OR discount_pct > 100", conn
        ).iloc[0]["cnt"]
        checks.append({
            "name"   : "valid_discount_range",
            "passed" : bad_disc == 0,
            "detail" : f"Geçersiz iskonto kayıtları: {bad_disc} (beklenen: 0)",
        })

        # ── Kontrol 5: Bölge çeşitliliği var mı? ────────────────────────────
        region_count = pd.read_sql(
            "SELECT COUNT(DISTINCT region) AS cnt FROM fact_sales", conn
        ).iloc[0]["cnt"]
        checks.append({
            "name"   : "region_diversity",
            "passed" : region_count >= 3,
            "detail" : f"Farklı bölge sayısı: {region_count} (min: 3)",
        })

    finally:
        conn.close()

    # Sonuçları raporla
    passed = sum(1 for c in checks if c["passed"])
    failed = len(checks) - passed

    ti.log_info(f"Kalite kontrol sonucu: {passed}/{len(checks)} geçti.")
    for c in checks:
        icon = "✓" if c["passed"] else "✗"
        ti.log_info(f"  {icon}  [{c['name']}] {c['detail']}")

    ti.xcom_push("quality_checks", checks)
    ti.xcom_push("quality_score",  round(passed / len(checks) * 100, 1))

    # Kritik kontrol başarısız → exception fırlat
    if failed > 0:
        failed_names = [c["name"] for c in checks if not c["passed"]]
        raise ValueError(f"Kalite kontrolü başarısız: {failed_names}")


# =============================================================================
# ❻  NOTIFY — Bildirim Görevi
# =============================================================================

def send_pipeline_notification(ti: TaskInstance, channel: str = "slack"):
    """
    [NOTIFY TASK] Pipeline tamamlandığında ekibi bilgilendir.
    
    Gerçek senaryoda:
        - Slack webhook: requests.post(SLACK_WEBHOOK_URL, json=payload)
        - E-posta: Airflow EmailOperator
        - PagerDuty: kritik hatalar için
        - Datadog/Grafana: metrik push
    """
    ti.log_info(f"Bildirim gönderiliyor ({channel})...")

    total_revenue = ti.xcom_pull("enrich_and_aggregate", "total_revenue")
    quality_score = ti.xcom_pull("run_data_quality_checks", "quality_score")
    removed_count = ti.xcom_pull("clean_and_validate_data", "removed_count")

    message = {
        "channel"      : f"#{channel}",
        "dag_id"       : ti.dag_id,
        "execution_date": ti.execution_date.strftime("%Y-%m-%d %H:%M"),
        "status"       : "✅ BAŞARILI",
        "metrics"      : {
            "toplam_ciro"         : f"₺{total_revenue:,.2f}" if total_revenue else "N/A",
            "kalite_skoru"        : f"%{quality_score}" if quality_score else "N/A",
            "temizlenen_kayitlar" : removed_count or 0,
        },
    }

    # Gerçekte: requests.post(webhook_url, json=message)
    ti.log_info(f"Bildirim içeriği: {json.dumps(message, ensure_ascii=False, indent=2)}")
    ti.log_info("Bildirim başarıyla gönderildi.")


# =============================================================================
# ❼  DAG TANIMLAMASI — Pipeline'ı Oluştur ve Bağla
# =============================================================================

def create_etl_dag() -> DAG:
    """
    ETL pipeline DAG'ını oluştur.
    
    Task bağımlılık grafiği:
    
        extract_sales ──┐
                        ├──► clean_validate ──► enrich_aggregate ──► load_dw ──► quality_check ──► notify
        extract_customers ──┘
    
    Gerçek Airflow'da bu fonksiyon modül düzeyinde çağrılır ve
    scheduler otomatik olarak DAG'ı keşfeder.
    """

    # Varsayılan task argümanları (tüm task'lara uygulanır)
    default_args = {
        "owner"           : "data-team",
        "depends_on_past" : False,         # Önceki çalışmanın bitmesini bekleme
        "email_on_failure": True,
        "email_on_retry"  : False,
    }

    dag = DAG(
        dag_id           ="etl_satis_pipeline",
        description      ="Satış verisi ETL pipeline'ı — günlük çalışır",
        schedule_interval="0 3 * * *",     # Her gün 03:00'da çalış
        start_date       =datetime(2024, 1, 1),
        catchup          =False,           # Geçmiş çalışmaları telafi etme
        tags             =["etl", "satis", "gunluk"],
        default_args     =default_args,
    )

    # ── Task Tanımlamaları ─────────────────────────────────────────────────
    t_extract_sales = PythonOperator(
        task_id          ="extract_sales_data",
        python_callable  =extract_sales_data,
        op_kwargs        ={"num_records": 500},
        retries          =2,
        group            ="extract",
    )

    t_extract_customers = PythonOperator(
        task_id         ="extract_customer_data",
        python_callable =extract_customer_data,
        retries         =2,
        group           ="extract",
    )

    t_clean = PythonOperator(
        task_id         ="clean_and_validate_data",
        python_callable =clean_and_validate_data,
        retries         =1,
        group           ="transform",
    )

    t_enrich = PythonOperator(
        task_id         ="enrich_and_aggregate",
        python_callable =enrich_and_aggregate,
        retries         =1,
        group           ="transform",
    )

    t_load = PythonOperator(
        task_id         ="load_to_data_warehouse",
        python_callable =load_to_data_warehouse,
        op_kwargs       ={"db_path": DEFAULT_DB_PATH},
        retries         =3,
        group           ="load",
    )

    t_quality = PythonOperator(
        task_id         ="run_data_quality_checks",
        python_callable =run_data_quality_checks,
        retries         =0,
        group           ="validate",
    )

    t_notify = PythonOperator(
        task_id         ="send_pipeline_notification",
        python_callable =send_pipeline_notification,
        op_kwargs       ={"channel": "data-alerts"},
        retries         =2,
        group           ="notify",
    )

    # ── Bağımlılıkları Tanımla (>> operatörü veya set_upstream) ───────────
    #
    # Airflow'da iki eşdeğer yazım vardır:
    #   Yöntem 1 (>> operatörü):   t_extract >> t_clean >> t_enrich
    #   Yöntem 2 (set_upstream):   t_clean.set_upstream(t_extract)
    #
    # Paralel extract task'ları temizleme öncesi birleşir:
    t_extract_sales    >> t_clean
    t_extract_customers >> t_clean
    t_clean >> t_enrich >> t_load >> t_quality >> t_notify

    # Tüm task'ları DAG'a kaydet
    for task in [
        t_extract_sales, t_extract_customers,
        t_clean, t_enrich, t_load, t_quality, t_notify,
    ]:
        dag.add_task(task)

    return dag


# =============================================================================
# ❽  ANA GİRİŞ NOKTASI
# =============================================================================

if __name__ == "__main__":
    """
    Bu blok sadece doğrudan `python etl_airflow_pipeline.py` ile
    çalıştırıldığında tetiklenir.
    
    Gerçek Airflow'da DAG dosyaları modül olarak import edilir;
    __main__ bloğuna gerek yoktur.
    
    Yerel geliştirme & test için bu simülasyon kullanışlıdır.
    """

    print("\n" + "█" * 70)
    print("  ETL / ELT Pipeline — Apache Airflow Mimarisi")
    print("  Satış Veri Ambarı Güncelleme Pipeline'ı")
    print("█" * 70)

    # DAG'ı oluştur
    satis_dag = create_etl_dag()

    # Manuel tetikleme (gerçekte: airflow dags trigger etl_satis_pipeline)
    results = satis_dag.run(execution_date=datetime.now())

    # Detaylı sonuç raporu
    print("\n📋  DETAYLI TASK SONUÇLARI")
    print("─" * 55)
    for task_id, ti in results.items():
        state_icon = {
            TaskState.SUCCESS: "✅",
            TaskState.FAILED : "❌",
            TaskState.SKIPPED: "⏭ ",
        }.get(ti.state, "❓")
        print(
            f"  {state_icon}  {task_id:<35}  "
            f"{ti.duration:.2f}s  [{ti.state.value}]"
        )

    # XCom özetini göster
    xcom_store = next(iter(results.values())).xcom
    print("\n🗂   XCOM DEĞERLERİ (Task'lar Arası Veri)")
    print("─" * 55)
    for key, value in xcom_store._store.items():
        display = value if not isinstance(value, str) or len(value) < 60 else value[:57] + "..."
        print(f"  {key:<45} → {display}")

    # Veritabanı özeti
    db_path = DEFAULT_DB_PATH
    if Path(db_path).exists():
        print(f"\n💾  VERİ AMBARI ÖZETI ({db_path})")
        print("─" * 55)
        conn = sqlite3.connect(db_path)
        for table in ["fact_sales", "dim_region_category", "fact_monthly_trend", "pipeline_runs"]:
            try:
                count = pd.read_sql(f"SELECT COUNT(*) AS c FROM {table}", conn).iloc[0]["c"]
                print(f"  Tablo: {table:<30} {count:>6} satır")
            except Exception:
                pass
        conn.close()

    print("\n✨  Pipeline simülasyonu tamamlandı.\n")
