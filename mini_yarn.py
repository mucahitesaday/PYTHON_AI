"""
=== MINI YARN (Yet Another Resource Negotiator) ==================================
YARN nedir? Küme kaynaklarını (CPU, RAM) yöneten ve uygulamaların bu kaynakları
kullanmasını sağlayan resource management katmanıdır.

Temel kavramlar:
- ResourceManager (RM): Kümenin patronu. Kaynakları tahsis eder, uygulamaları
  kabul eder/reddeder.
- NodeManager (NM): Her sunucuda çalışan ajan. Container'ları başlatır/izler.
- ApplicationMaster (AM): Her uygulamanın kendi yöneticisi. RM'den container
  ister, NM'lere görev dağıtır.
- Container: CPU/RAM gibi kaynakların bir birimi. İşler container içinde çalışır.
- Scheduler: RM içinde hangi uygulamaya ne kadar kaynak verileceğine karar verir.

Akış:
  1. Client, RM'ye uygulama gönderir
  2. RM bir container tahsis eder, ApplicationMaster başlatır
  3. AM, RM'den daha fazla container ister
  4. AM, container'ları NodeManager'lar üzerinde başlatır
  5. İş bitince AM kapanır, RM kaynakları geri alır
"""

import time
import uuid
import threading
from enum import Enum
from typing import Dict, List, Optional, Callable


# ==============================================================================
# 1. TEMEL VERİ YAPILARI
# ==============================================================================
class Resource:
    """CPU ve RAM'i temsil eden kaynak birimi."""
    def __init__(self, cpu: int = 1, memory_mb: int = 1024):
        self.cpu = cpu
        self.memory_mb = memory_mb

    def __repr__(self):
        return f"Resource(cpu={self.cpu}, mem={self.memory_mb}MB)"

    def __add__(self, other):
        return Resource(self.cpu + other.cpu, self.memory_mb + other.memory_mb)

    def __sub__(self, other):
        return Resource(self.cpu - other.cpu, self.memory_mb - other.memory_mb)

    def __ge__(self, other):
        return self.cpu >= other.cpu and self.memory_mb >= other.memory_mb

    def __le__(self, other):
        return self.cpu <= other.cpu and self.memory_mb <= other.memory_mb


class ContainerStatus(Enum):
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


class Container:
    """
    Container: Bir Node'da çalışan kaynak birimi.
    İçinde bir görev (task) çalıştırılır.
    """
    def __init__(self, container_id: str, node_id: str, resource: Resource):
        self.container_id = container_id
        self.node_id = node_id
        self.resource = resource
        self.status = ContainerStatus.RUNNING
        self.task = None
        self.result = None

    def run_task(self, task_func: Callable, *args):
        """Container içinde bir görev çalıştırır."""
        try:
            self.task = task_func
            self.result = task_func(*args)
            self.status = ContainerStatus.FINISHED
            return self.result
        except Exception as e:
            self.status = ContainerStatus.FAILED
            self.result = str(e)
            return None

    def __repr__(self):
        return f"Container({self.container_id} @ {self.node_id}, {self.status.value})"


# ==============================================================================
# 2. NODE MANAGER - Her sunucuda çalışan ajan
# ==============================================================================
class NodeManager:
    """
    NodeManager: Bir sunucuyu temsil eder.
    - Toplam kaynak (toplam CPU, RAM)
    - Kullanılabilir kaynak
    - Container'ları başlatma/yönetme
    """
    def __init__(self, node_id: str, total_resource: Resource):
        self.node_id = node_id
        self.total_resource = total_resource
        self.available_resource = total_resource
        self.containers: Dict[str, Container] = {}
        self.lock = threading.Lock()

    def can_allocate(self, resource: Resource) -> bool:
        """İstenen kaynak mevcut mu?"""
        with self.lock:
            return resource <= self.available_resource

    def allocate_container(self, container_id: str, resource: Resource) -> Optional[Container]:
        """
        Container tahsis eder.
        - Kaynak yeterliyse container oluştur, available_resource azalt
        """
        with self.lock:
            if not (resource <= self.available_resource):
                print(f"    [NM:{self.node_id}] Yetersiz kaynak! "
                      f"İstenen={resource}, Kullanılabilir={self.available_resource}")
                return None
            container = Container(container_id, self.node_id, resource)
            self.containers[container_id] = container
            self.available_resource = self.available_resource - resource
            print(f"    [NM:{self.node_id}] Container tahsis: {container_id} "
                  f"({resource}) Kalan: {self.available_resource}")
            return container

    def release_container(self, container_id: str):
        """Container'ı serbest bırakır, kaynakları geri alır."""
        with self.lock:
            container = self.containers.pop(container_id, None)
            if container:
                self.available_resource = self.available_resource + container.resource
                print(f"    [NM:{self.node_id}] Container serbest: {container_id} "
                      f"Kullanılabilir: {self.available_resource}")

    def get_available_resource(self) -> Resource:
        with self.lock:
            return self.available_resource

    def heartbeat(self) -> dict:
        """ResourceManager'a durum bildirir."""
        return {
            "node_id": self.node_id,
            "total": self.total_resource,
            "available": self.available_resource,
            "container_count": len(self.containers)
        }

    def __repr__(self):
        return f"NodeManager({self.node_id}, avail={self.available_resource})"


# ==============================================================================
# 3. APPLICATION MASTER - Her uygulamanın kendi yöneticisi
# ==============================================================================
class ApplicationStatus(Enum):
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


class ApplicationMaster:
    """
    ApplicationMaster: Her uygulama için ayrı bir yönetici.
    - RM'den container ister
    - Container'larda görev çalıştırır
    - İlerleme durumunu RM'ye bildirir
    """
    def __init__(self, app_id: str, user: str = "user"):
        self.app_id = app_id
        self.user = user
        self.status = ApplicationStatus.SUBMITTED
        self.containers: List[Container] = []
        self.tasks: List[dict] = []
        self.results: List = []

    def run(self, resource_manager, task_defs: List[dict]):
        """
        Uygulamayı çalıştırır.
        1. RM'den container iste
        2. Container'larda task'ları çalıştır
        3. Sonuçları topla
        """
        print(f"\n  [AM:{self.app_id}] Uygulama başlatılıyor...")
        self.status = ApplicationStatus.RUNNING

        for i, task_def in enumerate(task_defs):
            func = task_def.get("func")
            args = task_def.get("args", ())
            required_resource = task_def.get("resource", Resource(1, 512))

            # RM'den container iste
            container_id = f"container_{self.app_id}_{i:03d}"
            container = resource_manager.allocate_container(
                container_id, required_resource, self
            )

            if container is None:
                print(f"  [AM:{self.app_id}] Container tahsis edilemedi! "
                      f"Task {i} beklemeye alındı.")
                continue

            self.containers.append(container)

            # Container'da task'ı çalıştır
            print(f"  [AM:{self.app_id}] Task {i} çalıştırılıyor "
                  f"@ {container.node_id}...")
            result = container.run_task(func, *args)
            self.results.append(result)

            # Container'ı serbest bırak
            resource_manager.release_container(container_id)

        self.status = ApplicationStatus.FINISHED
        print(f"  [AM:{self.app_id}] Uygulama tamamlandı! "
              f"Sonuçlar: {self.results}")
        return self.results


# ==============================================================================
# 4. RESOURCE MANAGER - Kümenin patronu
# ==============================================================================
class ResourceManager:
    """
    ResourceManager: YARN'ın kalbi.
    - NodeManager'ları yönetir
    - Uygulamaları kabul eder
    - Container tahsisi yapar (scheduling)
    - Küme durumunu izler

    İki ana scheduler:
      FIFO: İlk gelen ilk çalışır (basit)
      Fair: Tüm uygulamalara eşit kaynak
    """
    def __init__(self, scheduler_type: str = "fair"):
        self.scheduler_type = scheduler_type
        self.nodemanagers: Dict[str, NodeManager] = {}
        self.applications: Dict[str, ApplicationMaster] = {}
        self.lock = threading.Lock()

    def register_nodemanager(self, nm: NodeManager):
        """Yeni bir NodeManager kaydeder."""
        with self.lock:
            self.nodemanagers[nm.node_id] = nm
            print(f"  [RM] NodeManager kaydedildi: {nm}")

    def submit_application(self, app_id: str, user: str = "user") -> ApplicationMaster:
        """Yeni bir uygulama gönderir."""
        am = ApplicationMaster(app_id, user)
        with self.lock:
            self.applications[app_id] = am
        print(f"  [RM] Uygulama alındı: {app_id} (user={user})")
        return am

    def allocate_container(self, container_id: str, resource: Resource,
                           app_master: ApplicationMaster) -> Optional[Container]:
        """
        En uygun NodeManager'da container tahsis eder.
        Scheduling stratejisine göre node seçer.
        """
        with self.lock:
            # Kullanılabilir NodeManager'ları bul
            candidates = []
            for nm in self.nodemanagers.values():
                if nm.can_allocate(resource):
                    candidates.append(nm)

            if not candidates:
                print(f"  [RM] UYARI: Container {container_id} için uygun node yok!")
                return None

            # Scheduler stratejisine göre seçim
            if self.scheduler_type == "fifo":
                # İlk uygun node'u seç
                chosen_nm = candidates[0]
            else:  # fair - en çok kaynağı olanı seç (load balancing)
                chosen_nm = max(candidates, key=lambda n: n.get_available_resource().cpu)

            print(f"  [RM] Container tahsis: {container_id} ({resource}) "
                  f"-> {chosen_nm.node_id} (scheduler={self.scheduler_type})")
            return chosen_nm.allocate_container(container_id, resource)

    def release_container(self, container_id: str):
        """Container'ı serbest bırakır."""
        for nm in self.nodemanagers.values():
            if container_id in nm.containers:
                nm.release_container(container_id)
                return

    def get_cluster_status(self) -> dict:
        """Küme durumunu döndürür."""
        with self.lock:
            total_cpu = sum(nm.total_resource.cpu for nm in self.nodemanagers.values())
            total_mem = sum(nm.total_resource.memory_mb for nm in self.nodemanagers.values())
            used_cpu = sum(
                nm.total_resource.cpu - nm.available_resource.cpu
                for nm in self.nodemanagers.values()
            )
            used_mem = sum(
                nm.total_resource.memory_mb - nm.available_resource.memory_mb
                for nm in self.nodemanagers.values()
            )
            return {
                "nodes": len(self.nodemanagers),
                "total_cpu": total_cpu,
                "total_mem": total_mem,
                "used_cpu": used_cpu,
                "used_mem": used_mem,
                "apps": len(self.applications),
                "apps_running": sum(
                    1 for a in self.applications.values()
                    if a.status == ApplicationStatus.RUNNING
                )
            }


# ==============================================================================
# 5. MINI YARN API
# ==============================================================================
class MiniYARN:
    """
    MiniYARN: Kullanıcıya sunulan üst düzey API.
    - NodeManager ekle
    - Uygulama gönder
    - Küme durumunu izle
    """
    def __init__(self, scheduler_type: str = "fair"):
        self.rm = ResourceManager(scheduler_type)

    def add_node(self, node_id: str = None, cpu: int = 4, memory_mb: int = 8192) -> NodeManager:
        """Yeni bir iş düğümü ekler."""
        if node_id is None:
            node_id = f"node-{uuid.uuid4().hex[:4]}"
        nm = NodeManager(node_id, Resource(cpu, memory_mb))
        self.rm.register_nodemanager(nm)
        return nm

    def run_application(self, app_id: str, tasks: List[dict]) -> List:
        """Bir uygulama gönderir ve çalıştırır."""
        am = self.rm.submit_application(app_id)
        return am.run(self.rm, tasks)

    def status(self):
        """Küme durumunu gösterir."""
        s = self.rm.get_cluster_status()
        print(f"\n{'='*60}")
        print(f"MINI YARN STATUS (scheduler={self.rm.scheduler_type})")
        print(f"{'='*60}")
        print(f"Düğümler: {s['nodes']}")
        print(f"Toplam Kaynak: CPU={s['total_cpu']}, RAM={s['total_mem']}MB")
        print(f"Kullanılan: CPU={s['used_cpu']}, RAM={s['used_mem']}MB")
        print(f"Uygulamalar: Toplam={s['apps']}, Çalışan={s['apps_running']}")
        for nm in self.rm.nodemanagers.values():
            hb = nm.heartbeat()
            print(f"  {nm.node_id}: {hb['available']} / {nm.total_resource}")
        print(f"{'='*60}\n")


# ==============================================================================
# 6. TEST / DEMO
# ==============================================================================
def test_mini_yarn():
    """
    Mini YARN testi:
    1. 2 NodeManager ekle
    2. WordCount benzeri 3 task gönder
    3. Sonuçları göster
    """
    print("\n" + "="*60)
    print("MINI YARN DEMO")
    print("="*60)

    # YARN kurulum
    yarn = MiniYARN(scheduler_type="fair")

    # Node'ları ekle
    yarn.add_node("worker-1", cpu=4, memory_mb=8192)
    yarn.add_node("worker-2", cpu=8, memory_mb=16384)
    yarn.add_node("worker-3", cpu=2, memory_mb=4096)

    # Task fonksiyonları
    def count_words(text: str) -> dict:
        """Bir metindeki kelimeleri sayar."""
        words = text.lower().split()
        counts = {}
        for w in words:
            w = w.strip(".,!?;:")
            counts[w] = counts.get(w, 0) + 1
        print(f"      [Task] Kelime sayımı tamam: {len(counts)} unique kelime")
        return counts

    def merge_counts(counts_list: List[dict]) -> dict:
        """Birden fazla sayım sonucunu birleştirir."""
        merged = {}
        for c in counts_list:
            for word, count in c.items():
                merged[word] = merged.get(word, 0) + count
        return merged

    def format_result(counts: dict) -> str:
        """Sonucu formatlar."""
        sorted_words = sorted(counts.items(), key=lambda x: -x[1])[:10]
        return "\n".join(f"    {w}: {c}" for w, c in sorted_words)

    # Task tanımları
    tasks = [
        {"func": count_words, "args": ("hadoop yarn spark bigdata distributed computing",),
         "resource": Resource(1, 512)},
        {"func": count_words, "args": ("yarn manages resources spark processes data hdfs stores",),
         "resource": Resource(1, 512)},
        {"func": count_words, "args": ("distributed systems are powerful and scalable",),
         "resource": Resource(1, 512)},
    ]

    # Uygulamayı çalıştır
    results = yarn.run_application("app-wordcount", tasks)

    # Sonuçları birleştir
    print(f"\n  [Main] Sonuçlar birleştiriliyor...")
    merged = merge_counts(results)
    print(f"\n  [Main] TOP 10 KELIME:")
    print(format_result(merged))

    # Sistem durumu
    yarn.status()

    return yarn


if __name__ == "__main__":
    test_mini_yarn()
