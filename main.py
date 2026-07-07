"""
===============================================================================
BÜYÜK VERİ SİSTEMLERİ - EĞİTİM AMAÇLI MİNİ SİMÜLASYONLAR
===============================================================================
Bu proje 3 ana big data bileşenini basitçe simüle eder:

  1. MINI HDFS  -> Distributed File System (Dağıtık Dosya Sistemi)
  2. MINI YARN  -> Resource Manager (Kaynak Yöneticisi)
  3. MINI SPARK -> Data Processing Engine (Veri İşleme Motoru)

Her biri ayrı bir .py dosyasında, Türkçe açıklamalı ve çalışır durumdadır.
===============================================================================

GERÇEK HAYATTA AKIŞ:
  1. HDFS'ye veri yazılır (dosyalar block'lara bölünür, DataNode'lara dağıtılır)
  2. YARN kümesi kaynakları yönetir (hangi node'da ne kadar CPU/RAM var)
  3. Spark, HDFS'den veriyi okur, YARN'dan container ister, işlemi yapar

Bu simülasyon her üç sistemi ayrı ayrı gösterir.
"""

import sys
import os

# Renkli çıktı için (opsiyonel)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.END}")
    print(f"{Colors.HEADER}{'='*60}{Colors.END}")


def print_step(num: int, text: str):
    print(f"\n{Colors.CYAN}>>> ADIM {num}: {text}{Colors.END}")


def show_menu():
    print(f"""
{Colors.BOLD}{'='*60}{Colors.END}
{Colors.BOLD}  BÜYÜK VERİ SİSTEMLERİ - MİNİ SİMÜLASYON{Colors.END}
{Colors.BOLD}{'='*60}{Colors.END}

  Bu program 3 büyük veri bileşenini eğitim amaçlı simüle eder:

  {Colors.GREEN}[1] MINI HDFS{Colors.END}
      -> Dağıtık Dosya Sistemi
      -> Dosyayı block'lara böler, DataNode'larda replike eder
      -> Fault tolerance gösterir (node düşse bile veriye erişim)

  {Colors.GREEN}[2] MINI YARN{Colors.END}
      -> Küme Kaynak Yöneticisi
      -> CPU/RAM yönetimi, container tahsisi
      -> Uygulamaları Node'lara dağıtır

  {Colors.GREEN}[3] MINI SPARK{Colors.END}
      -> Dağıtık Veri İşleme Motoru
      -> RDD (Resilient Distributed Dataset)
      -> Lazy transformations + Action tetikleme
      -> WordCount örneği

  {Colors.GREEN}[4] TÜMÜNÜ ÇALIŞTIR{Colors.END}
  {Colors.GREEN}[0] ÇIKIŞ{Colors.END}
""")


def run_hdfs_demo():
    from mini_dfs import test_mini_hdfs
    print_header("BÖLÜM 1: MINI HDFS - Dağıtık Dosya Sistemi")

    print(f"""
{Colors.BOLD}HDFS (Hadoop Distributed File System){Colors.END}
{Colors.BOLD}{'-'*50}{Colors.END}
HDFS, büyük dosyaları (GB/TB) block'lara bölüp birden çok sunucuda
depolar. Varsayılan block boyutu 128MB'dir. Her block 3 kopya halinde
saklanır (replication factor = 3). Böylece bir sunucu çökse bile
veri kaybolmaz.

Bu simülasyonda:
  - NameNode: metadata'yı yönetir (hangi dosya hangi block'lardan oluşur)
  - DataNode: block'ları depolar
  - Replication: her block birden fazla DataNode'da saklanır
  - Fault Tolerance: bir DataNode düşse bile veri okunabilir
{Colors.BOLD}{'-'*50}{Colors.END}
""")
    result = test_mini_hdfs()
    print(f"\n{Colors.GREEN}HDFS Demo başarıyla tamamlandı!{Colors.END}")
    return result


def run_yarn_demo():
    from mini_yarn import test_mini_yarn
    print_header("BÖLÜM 2: MINI YARN - Küme Kaynak Yöneticisi")

    print(f"""
{Colors.BOLD}YARN (Yet Another Resource Negotiator){Colors.END}
{Colors.BOLD}{'-'*50}{Colors.END}
YARN, bir kümedeki CPU ve RAM gibi kaynakları yönetir.
Uygulamalar YARN'a gönderilir, YARN da uygun node'larda
container'lar açar ve uygulamayı çalıştırır.

Bileşenler:
  - ResourceManager (RM): Kümenin patronu, kaynakları tahsis eder
  - NodeManager (NM): Her sunucuda çalışan ajan, container'ları yönetir
  - ApplicationMaster (AM): Her uygulamanın kendi yöneticisi
  - Container: CPU/RAM birimi, işler burada çalışır
{Colors.BOLD}{'-'*50}{Colors.END}
""")
    result = test_mini_yarn()
    print(f"\n{Colors.GREEN}YARN Demo başarıyla tamamlandı!{Colors.END}")
    return result


def run_spark_demo():
    from mini_spark import test_mini_spark
    print_header("BÖLÜM 3: MINI SPARK - Veri İşleme Motoru")

    print(f"""
{Colors.BOLD}Apache Spark - Distributed Data Processing{Colors.END}
{Colors.BOLD}{'-'*50}{Colors.END}
Spark, büyük veriyi hafızada (in-memory) işler. MapReduce'tan
100 kata kadar daha hızlıdır.

Temel Kavramlar:
  - RDD: Partition'lara bölünmüş, dağıtık veri
  - Transformation (lazy): map, filter, flatMap, reduceByKey
  - Action (eager): collect, count, reduce (tetikler)
  - Lineage: RDD'nin soy ağacı (kayıp durumunda yeniden hesaplama)
  - Shuffle: Verinin partition'lar arasında taşınması

Örnek: WordCount
  lines.flatMap(split) -> map(word,1) -> reduceByKey(+) -> collect
{Colors.BOLD}{'-'*50}{Colors.END}
""")
    result = test_mini_spark()
    print(f"\n{Colors.GREEN}Spark Demo başarıyla tamamlandı!{Colors.END}")
    return result


def run_all():
    print_header("TÜM BİLEŞENLER ÇALIŞTIRILIYOR")
    print(f"""
{Colors.BOLD}GERÇEK HAYATTA BU 3 SİSTEM ŞÖYLE ÇALIŞIR:{Colors.END}

  1. Veri HDFS'ye yazılır
     -> Büyük dosya block'lara bölünür
     -> Block'lar DataNode'lara dağıtılır (replication ile)

  2. Spark uygulaması YARN'a gönderilir
     -> ResourceManager kaynakları tahsis eder
     -> NodeManager'larda container'lar açılır
     -> ApplicationMaster işi yönetir

  3. Spark HDFS'den veriyi okur ve işler
     -> RDD'ler oluşturulur
     -> Transformations ile DAG kurulur
     -> Action ile hesaplama tetiklenir
     -> Sonuç driver'a döner

{Colors.BOLD}{'-'*50}{Colors.END}
""")

    run_hdfs_demo()
    run_yarn_demo()
    run_spark_demo()

    print_header("TÜM DEMOLAR TAMAMLANDI!")
    print(f"""
{Colors.GREEN}{'='*60}{Colors.END}
{Colors.GREEN}  Başarıyla tamamlandı! Her bir .py dosyasını ayrı ayrı{Colors.END}
{Colors.GREEN}  inceleyerek kodları detaylıca görebilirsiniz.{Colors.END}
{Colors.GREEN}{'='*60}{Colors.END}

{Colors.BOLD}Dosyalar:{Colors.END}
  - mini_dfs.py   : HDFS simülasyonu (NameNode, DataNode, replication)
  - mini_yarn.py  : YARN simülasyonu (ResourceManager, NodeManager)
  - mini_spark.py : Spark simülasyonu (RDD, DAG, transformations)
  - main.py       : Bu dosya (menü ve entegrasyon)

{Colors.BOLD}Önerilen inceleme sırası:{Colors.END}
  1. Önce mini_dfs.py (veri depolama)
  2. Sonra mini_yarn.py (kaynak yönetimi)
  3. En son mini_spark.py (veri işleme)
""")


def main():
    while True:
        show_menu()
        choice = input(f"{Colors.BOLD}Seçiminiz (0-4): {Colors.END}").strip()

        if choice == "1":
            run_hdfs_demo()
            input(f"\n{Colors.WARNING}Devam etmek için Enter'a basın...{Colors.END}")

        elif choice == "2":
            run_yarn_demo()
            input(f"\n{Colors.WARNING}Devam etmek için Enter'a basın...{Colors.END}")

        elif choice == "3":
            run_spark_demo()
            input(f"\n{Colors.WARNING}Devam etmek için Enter'a basın...{Colors.END}")

        elif choice == "4":
            run_all()
            input(f"\n{Colors.WARNING}Devam etmek için Enter'a basın...{Colors.END}")

        elif choice == "0":
            print(f"\n{Colors.GREEN}Program sonlandırıldı. İyi çalışmalar!{Colors.END}")
            break

        else:
            print(f"\n{Colors.FAIL}Geçersiz seçim! Lütfen 0-4 arası bir değer girin.{Colors.END}")


if __name__ == "__main__":
    # Windows'ta renk desteği
    if os.name == 'nt':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            # Renk desteklenmiyorsa devre dışı bırak
            for attr in dir(Colors):
                if not attr.startswith('_'):
                    setattr(Colors, attr, '')

    main()
