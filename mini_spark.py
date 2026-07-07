"""
=== MINI SPARK (Distributed Data Processing Engine) =============================
Spark nedir? Büyük veriyi hafızada (in-memory) işleyen, dağıtık hesaplama
motorudur. MapReduce'tan ~100x daha hızlıdır çünkü ara sonuçları diske yazmak
yerine hafızada tutar.

Temel kavramlar:
- RDD (Resilient Distributed Dataset): Bölünemez, dağıtık veri koleksiyonu.
  Temel soyutlama. Partition'lara bölünür ve cluster'da dağıtılır.
  Lineage (soy ağacı) sayesinde bir partition kaybolursa yeniden hesaplanabilir.
- Transformation: RDD'den yeni RDD üreten lazy işlemler (map, filter, flatMap).
  Hesaplanmaz, sadece DAG'e eklenir.
- Action: Sonuç döndüren eager işlemler (collect, count, reduce).
  DAG'in çalıştırılmasını tetikler.
- DAG (Directed Acyclic Graph): Yönlendirilmiş asiklik graf.
  RDD'ler arasındaki bağımlılıkları gösterir.
- Stage: DAG'in shuffle sınırlarıyla bölünmüş parçası.
- Task: Stage'in tek bir partition üzerinde çalışan en küçük birimi.
- Shuffle: Verinin partition'lar arasında yeniden dağıtılması (ağ trafiği).

Akış:
  RDD (okuma) -> Transformation (map/filter) -> Transformation (reduceByKey)
  -> Action (collect) tetikler -> DAG Scheduler -> Stage'ler -> Task'ler
  -> Worker'lar
"""

import threading
from typing import (Any, Callable, Dict, Generic, List, Optional,
                    Set, Tuple, TypeVar)

T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")
V = TypeVar("V")


# ==============================================================================
# 1. PARTITION - Verinin en küçük birimi
# ==============================================================================
class Partition(Generic[T]):
    """
    Partition: RDD'nin bölündüğü veri parçası.
    Her partition farklı bir node'da işlenir (paralellik).
    """
    def __init__(self, index: int, data: List[T]):
        self.index = index
        self.data = data

    def __repr__(self):
        return f"Partition({self.index}, {len(self.data)} eleman)"


# ==============================================================================
# 2. RDD - Resilient Distributed Dataset (TEMEL SOYUTLAMA)
# ==============================================================================
class RDD(Generic[T]):
    """
    RDD: Spark'ın temel veri yapısı.
    - partition'lara bölünmüş veri
    - lineage (soy ağacı) - nasıl oluştuğunu bilir
    - lazy - sadece action çağrılınca hesaplanır
    - immutable - bir kere oluşunca değişmez, yeni RDD üretilir

    Operasyon tipleri:
      Transformation (lazy): map, filter, flatMap, reduceByKey -> yeni RDD
      Action (eager):       collect, count, reduce, foreach -> sonuç
    """
    def __init__(self, partitions: Optional[List[Partition]] = None,
                 compute_func: Optional[Callable] = None,
                 dependencies: Optional[List["RDD"]] = None,
                 name: str = "anon"):
        self.partitions_data = partitions
        self.compute_func = compute_func
        self.dependencies = dependencies or []
        self.name = name

    def get_partitions(self) -> List[Partition]:
        """Partition'ları hesapla/ getir."""
        if self.compute_func and not self.partitions_data:
            self.partitions_data = self.compute_func()
        return self.partitions_data or []

    # ===== TRANSFORMATIONS (Lazy - hemen hesaplanmaz) =====

    def map(self, func: Callable[[T], U]) -> "RDD[U]":
        """
        map: Her elemana bir fonksiyon uygular.
        Örnek: rdd.map(lambda x: x * 2) -> her sayıyı 2 ile çarpar
        """
        def compute():
            partitions = self.get_partitions()
            new_parts = []
            for p in partitions:
                new_data = []
                for item in p.data:
                    new_data.append(func(item))
                new_parts.append(Partition(p.index, new_data))
            return new_parts

        return RDD(compute_func=compute, dependencies=[self],
                   name=f"map({self.name})")

    def filter(self, predicate: Callable[[T], bool]) -> "RDD[T]":
        """
        filter: Sadece koşulu sağlayan elemanları tutar.
        Örnek: rdd.filter(lambda x: x > 10) -> 10'dan büyükler
        """
        def compute():
            partitions = self.get_partitions()
            new_parts = []
            for p in partitions:
                new_data = [item for item in p.data if predicate(item)]
                new_parts.append(Partition(p.index, new_data))
            return new_parts

        return RDD(compute_func=compute, dependencies=[self],
                   name=f"filter({self.name})")

    def flatMap(self, func: Callable[[T], List[U]]) -> "RDD[U]":
        """
        flatMap: Her elemandan 0..n eleman üretir ve düzleştirir.
        Örnek: rdd.flatMap(lambda x: x.split()) -> cümleleri kelimelere ayırır
        """
        def compute():
            partitions = self.get_partitions()
            new_parts = []
            for p in partitions:
                new_data = []
                for item in p.data:
                    new_data.extend(func(item))
                new_parts.append(Partition(p.index, new_data))
            return new_parts

        return RDD(compute_func=compute, dependencies=[self],
                   name=f"flatMap({self.name})")

    def reduceByKey(self, func: Callable[[V, V], V]) -> "RDD[Tuple[K, V]]":
        """
        reduceByKey: Anahtarlara göre gruplar ve birleştirir.
        Sadece (key, value) çiftleri için çalışır.
        Bu işlem SHUFFLE gerektirir: veri partition'lar arasında taşınır.
        """
        def compute():
            partitions = self.get_partitions()

            # 1. AŞAMA: Her partition'da local reduce
            local_results = []
            for p in partitions:
                local: Dict[K, V] = {}
                for k, v in p.data:
                    if k in local:
                        local[k] = func(local[k], v)
                    else:
                        local[k] = v
                local_results.append(local)

            # 2. AŞAMA: SHUFFLE - anahtarları partition'lar arasında
            # yeniden dağıt (gerçek Spark'da ağ üzerinden)
            all_keys: Set[K] = set()
            for lr in local_results:
                all_keys.update(lr.keys())

            # Anahtarları hash ile partition'lara dağıt
            num_parts = len(partitions)
            shuffled: List[Dict[K, V]] = [{} for _ in range(num_parts)]
            for key in all_keys:
                target_part = hash(key) % num_parts
                for lr in local_results:
                    if key in lr:
                        if key in shuffled[target_part]:
                            shuffled[target_part][key] = func(
                                shuffled[target_part][key], lr[key])
                        else:
                            shuffled[target_part][key] = lr[key]

            new_parts = []
            for i, sd in enumerate(shuffled):
                data = list(sd.items())
                new_parts.append(Partition(i, data))
            return new_parts

        return RDD(compute_func=compute, dependencies=[self],
                   name=f"reduceByKey({self.name})")

    # ===== ACTIONS (Eager - hesaplamayı TETIKLER) =====

    def collect(self) -> List[T]:
        """
        collect: Tüm veriyi driver'a toplar.
        Büyük veride dikkat! (driver hafızası yetmeyebilir)
        """
        print(f"  [Spark] Action: collect tetiklendi")
        result = []
        partitions = self.get_partitions()
        for p in partitions:
            result.extend(p.data)
        return result

    def count(self) -> int:
        """
        count: Eleman sayısını döndürür.
        """
        print(f"  [Spark] Action: count tetiklendi")
        total = 0
        for p in self.get_partitions():
            total += len(p.data)
        return total

    def reduce(self, func: Callable[[T, T], T]) -> Optional[T]:
        """
        reduce: Tüm elemanları bir fonksiyonla birleştirir.
        Örnek: rdd.reduce(lambda a,b: a+b) -> toplam
        """
        print(f"  [Spark] Action: reduce tetiklendi")
        partitions = self.get_partitions()
        if not partitions or not partitions[0].data:
            return None
        result = partitions[0].data[0]
        for p in partitions:
            for item in p.data[1:]:
                result = func(result, item)
        return result

    def foreach(self, func: Callable[[T], None]):
        """
        foreach: Her eleman için bir fonksiyon çağırır (side-effect).
        """
        print(f"  [Spark] Action: foreach tetiklendi")
        for p in self.get_partitions():
            for item in p.data:
                func(item)

    def take(self, n: int) -> List[T]:
        """
        take: İlk n elemanı döndürür.
        """
        print(f"  [Spark] Action: take({n}) tetiklendi")
        result = []
        for p in self.get_partitions():
            for item in p.data:
                result.append(item)
                if len(result) >= n:
                    return result[:n]
        return result

    def show_lineage(self, indent: str = ""):
        """
        Lineage (soy ağacı) gösterimi.
        RDD'nin nasıl oluştuğunu adım adım gösterir.
        """
        print(f"{indent}RDD({self.name})")
        for dep in self.dependencies:
            dep.show_lineage(indent + "  ")

    def __repr__(self):
        return f"RDD({self.name})"


# ==============================================================================
# 3. SPARK CONTEXT - Spark uygulamasının giriş noktası
# ==============================================================================
class SparkContext:
    """
    SparkContext: Spark uygulamasının ana giriş noktası.
    - RDD oluşturma (parallelize, textFile)
    - Konfigürasyon (partition sayısı)
    - RDD lineage görüntüleme
    """
    def __init__(self, app_name: str = "MiniSparkApp",
                 num_partitions: int = 2):
        self.app_name = app_name
        self.num_partitions = num_partitions
        print(f"  [SparkContext] Başlatıldı: {app_name}, "
              f"partition={num_partitions}")

    def parallelize(self, data: List[T]) -> RDD[T]:
        """
        parallelize: Python listesinden RDD oluşturur.
        Veriyi partition'lara böler.
        """
        partitions = []
        n = len(data)
        part_size = max(1, n // self.num_partitions)
        for i in range(self.num_partitions):
            start = i * part_size
            end = start + part_size if i < self.num_partitions - 1 else n
            chunk = data[start:end]
            partitions.append(Partition(i, chunk))
        return RDD(partitions=partitions, name="parallelize")

    def text_file(self, lines: List[str], filename: str = "file") -> RDD[str]:
        """
        textFile: Metin satırlarından RDD oluşturur.
        Gerçek Spark'da HDFS'den okur.
        """
        return self.parallelize(lines)

    def set_log_level(self, level: str):
        print(f"  [SparkContext] Log seviyesi: {level}")

    def stop(self):
        print(f"  [SparkContext] Durduruldu: {self.app_name}")


# ==============================================================================
# 4. DAG GÖSTERİMİ
# ==============================================================================
class DAGVisualizer:
    """
    DAG: RDD'ler arasındaki bağımlılıkları gösteren yönlendirilmiş asiklik graf.
    - Her ok bir transformation'ı temsil eder
    - Mavi: transformation (lazy)
    - Kırmızı: shuffle (reduceByKey gibi)
    - Yeşil: action (tetikleyici)
    """
    @staticmethod
    def show(rdds: List[RDD], actions: List[str]):
        print(f"\n  {'='*55}")
        print(f"  DAG (Directed Acyclic Graph) - İşlem Akışı")
        print(f"  {'='*55}")
        for i, rdd in enumerate(rdds):
            prefix = "  " * (len(rdd.dependencies) + 1)
            print(f"  {prefix}└─ {rdd}")
        print(f"  {'='*55}")
        print(f"  Actions: {', '.join(actions)}")
        print(f"  {'='*55}\n")


# ==============================================================================
# 5. TEST / DEMO
# ==============================================================================
def test_mini_spark():
    """
    Mini Spark testi - WordCount örneği:
    1. SparkContext oluştur
    2. Metin satırlarından RDD oluştur
    3. flatMap -> map -> reduceByKey (transformation'lar)
    4. collect (action) -> sonucu göster
    5. Lineage göster
    """
    print("\n" + "="*60)
    print("MINI SPARK DEMO - WordCount")
    print("="*60)

    # SparkContext oluştur
    sc = SparkContext("WordCount", num_partitions=3)

    # Veri (HDFS'den okunmuş gibi)
    lines = [
        "hadoop yarn spark bigdata",
        "spark processes data fast",
        "yarn manages cluster resources",
        "hdfs stores data reliably",
        "spark is faster than mapreduce",
        "bigdata tools are powerful",
    ]

    # WordCount pipeline (transformation'lar)
    print(f"\n  WordCount Pipeline kuruluyor...")
    rdd_lines = sc.text_file(lines, "hdfs://data.txt")
    rdd_words = rdd_lines.flatMap(lambda line: line.lower().split())
    rdd_pairs = rdd_words.map(lambda word: (word, 1))
    rdd_counts = rdd_pairs.reduceByKey(lambda a, b: a + b)

    # Lineage göster
    print(f"\n  RDD Lineage (Soy Ağacı):")
    rdd_counts.show_lineage()

    # DAG görselleştirme
    DAGVisualizer.show(
        [rdd_lines, rdd_words, rdd_pairs, rdd_counts],
        ["collect"]
    )

    # Action - hesaplamayı tetikle
    print(f"  Action tetikleniyor: collect...")
    result = rdd_counts.collect()

    # Sonuçları göster
    sorted_result = sorted(result, key=lambda x: -x[1])
    print(f"\n  SONUÇ (Kelime Frekansları):")
    print(f"  {'='*40}")
    for word, count in sorted_result:
        print(f"    {word:15s}: {count}")
    print(f"  {'='*40}")
    print(f"  Toplam {len(sorted_result)} farklı kelime")

    sc.stop()
    return sc


if __name__ == "__main__":
    test_mini_spark()
