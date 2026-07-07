"""
=== MINI HDFS (Distributed File System) ========================================
HDFS nedir? Büyük dosyaları parçalara (block) bölüp birden fazla sunucuda
saklayan dosya sistemidir. Burada local'de thread'lerle simüle ediyoruz.

Temel kavramlar:
- NameNode: Dosyaların hangi block'lardan oluştuğunu ve block'ların hangi
  DataNode'larda olduğunu tutan merkezi yapı.
- DataNode: Block'ları fiziksel olarak depolayan düğüm.
- Block: Dosyanın bölündüğü sabit boyutlu parçalar (gerçek HDFS'de 128MB).
- Replication: Her block'un birden fazla kopyasının farklı node'larda
  tutulması (güvenilirlik için).

Kod yapısı:
  DataNode     -> Block'ları depolayan sunucu
  NameNode     -> Metadata'yı yöneten merkezi sunucu
  MiniHDFS     -> Kullanıcının kullandığı üst düzey API (Client)
  test_mini_hdfs -> Test fonksiyonu
"""

import threading
import uuid
import os
from typing import Dict, List, Optional

# ==============================================================================
# 1. DATANODE - Block'ları fiziksel olarak saklar
# ==============================================================================
class DataNode:
    """
    DataNode: Her block'u hafızada (gerçekte diskte) tutar.
    - block_id -> bytes (block verisi)
    - NameNode'dan gelen store/retrieve komutlarını çalıştırır.
    - Periyodik olarak NameNode'a heartbeat gönderir (canlı olduğunu bildirir).
    """
    def __init__(self, node_id: str, address: str):
        self.node_id = node_id
        self.address = address
        self.storage: Dict[str, bytes] = {}  # block_id -> içerik
        self.lock = threading.Lock()
        self.alive = True

    def store_block(self, block_id: str, data: bytes) -> bool:
        """Bir block'u depolar."""
        with self.lock:
            self.storage[block_id] = data
            return True

    def retrieve_block(self, block_id: str) -> Optional[bytes]:
        """Bir block'u getirir."""
        with self.lock:
            return self.storage.get(block_id)

    def delete_block(self, block_id: str) -> bool:
        """Bir block'u siler."""
        with self.lock:
            return self.storage.pop(block_id, None) is not None

    def get_block_ids(self) -> List[str]:
        """Saklanan tüm block ID'lerini döndürür."""
        with self.lock:
            return list(self.storage.keys())

    def heartbeat(self) -> bool:
        """NameNode'a canlı olduğunu bildirir (simülasyon)."""
        return self.alive

    def __repr__(self):
        return f"DataNode({self.node_id}, {len(self.storage)} blocks)"


# ==============================================================================
# 2. NAMENODE - Metadata'yı yöneten merkezi sunucu
# ==============================================================================
class NameNode:
    """
    NameNode: Dosya sisteminin "beyni".
    - Dosya adı -> hangi block'lardan oluştuğu
    - Her block -> hangi DataNode'larda olduğu (replication)
    - Block boyutu, replication faktörü gibi konfigürasyonlar
    """
    def __init__(self, replication_factor: int = 2, block_size: int = 1024):
        self.replication_factor = replication_factor
        self.block_size = block_size
        self.file_to_blocks: Dict[str, List[str]] = {}    # filename -> [block_id, ...]
        self.block_to_datanodes: Dict[str, List[str]] = {} # block_id -> [node_id, ...]
        self.datanodes: Dict[str, DataNode] = {}           # node_id -> DataNode
        self.next_block_id = 0
        self.lock = threading.Lock()

    def register_datanode(self, datanode: DataNode):
        """Yeni bir DataNode kaydeder."""
        with self.lock:
            self.datanodes[datanode.node_id] = datanode
            print(f"  [NameNode] DataNode kaydedildi: {datanode.node_id}")

    def get_available_datanodes(self) -> List[DataNode]:
        """Canlı DataNode'ları döndürür."""
        return [dn for dn in self.datanodes.values() if dn.heartbeat()]

    def _generate_block_id(self) -> str:
        """Her block'a unique ID üretir."""
        bid = f"block_{self.next_block_id:04d}"
        self.next_block_id += 1
        return bid

    def _select_target_nodes(self) -> List[DataNode]:
        """
        Block kopyalarını koymak için DataNode seçer.
        Replication faktörü kadar node seçer, rack-aware mantığı basitleştirilmiş.
        """
        available = self.get_available_datanodes()
        if len(available) < self.replication_factor:
            print(f"  [NameNode] UYARI: Yeterli DataNode yok! "
                  f"İstenen={self.replication_factor}, Var={len(available)}")
            return available  # Ne kadar varsa o kadar
        # İlk replication_factor kadar node'u seç
        return available[:self.replication_factor]

    def store_file(self, filename: str, data: bytes) -> bool:
        """
        Bir dosyayı block'lara bölüp DataNode'lara dağıtır.
        Adımlar:
          1. DataNode'ların canlı olduğunu kontrol et
          2. Dosyayı block_size kadar parçalara böl
          3. Her parça için: block_id oluştur, uygun DataNode'ları seç,
             kopyaları depola, metadata'yı güncelle
        """
        print(f"\n  [NameNode] Dosya depolanıyor: {filename} "
              f"(boyut={len(data)} bytes, block_size={self.block_size})")
        with self.lock:
            available_dns = self.get_available_datanodes()
            if not available_dns:
                print("  [NameNode] HATA: Hiç DataNode yok!")
                return False

            block_ids = []
            offset = 0
            while offset < len(data):
                # Dosyayı block_size kadar parçala
                chunk = data[offset:offset + self.block_size]
                block_id = self._generate_block_id()

                # Bu block'un kopyalarını koyacak DataNode'ları seç
                target_nodes = self._select_target_nodes()

                # Her hedef DataNode'a block'u yaz
                stored_nodes = []
                for dn in target_nodes:
                    dn.store_block(block_id, chunk)
                    stored_nodes.append(dn.node_id)

                # Metadata'yı güncelle
                self.block_to_datanodes[block_id] = stored_nodes
                block_ids.append(block_id)

                print(f"    Block {block_id}: {len(chunk)} bytes -> "
                      f"{stored_nodes} (replication={len(stored_nodes)})")
                offset += self.block_size

            self.file_to_blocks[filename] = block_ids
            print(f"  [NameNode] Dosya başarıyla depolandı: {filename} "
                  f"({len(block_ids)} blocks)")
            return True

    def retrieve_file(self, filename: str) -> Optional[bytes]:
        """
        Bir dosyayı block'lardan birleştirerek okur.
        Adımlar:
          1. Dosyanın block listesini metadata'dan al
          2. Her block için DataNode'lardan birinden oku (replication sayesinde
             bir node düşse bile diğerinden okunabilir)
          3. Block'ları sırayla birleştirip döndür
        """
        print(f"\n  [NameNode] Dosya okunuyor: {filename}")
        with self.lock:
            if filename not in self.file_to_blocks:
                print(f"  [NameNode] HATA: Dosya bulunamadı: {filename}")
                return None

            block_ids = self.file_to_blocks[filename]
            result = bytearray()

            for block_id in block_ids:
                # Bu block hangi DataNode'larda?
                node_ids = self.block_to_datanodes.get(block_id, [])
                block_data = None

                # Herhangi bir canlı DataNode'dan block'u al
                for node_id in node_ids:
                    dn = self.datanodes.get(node_id)
                    if dn and dn.heartbeat():
                        block_data = dn.retrieve_block(block_id)
                        if block_data is not None:
                            print(f"    Block {block_id} okundu: {node_id}")
                            break

                if block_data is None:
                    print(f"  [NameNode] HATA: Block {block_id} okunamadı! "
                          f"(Tüm replicas ölü)")
                    return None

                result.extend(block_data)

            print(f"  [NameNode] Dosya başarıyla okundu: {filename} "
                  f"({len(result)} bytes)")
            return bytes(result)

    def list_files(self) -> List[str]:
        """Dosya sistemindeki tüm dosyaları listeler."""
        with self.lock:
            return list(self.file_to_blocks.keys())

    def get_file_blocks(self, filename: str) -> Optional[List[str]]:
        """Bir dosyanın block ID'lerini döndürür (client için)."""
        with self.lock:
            return self.file_to_blocks.get(filename)

    def get_block_locations(self, block_id: str) -> List[str]:
        """Bir block'un hangi DataNode'larda olduğunu döndürür."""
        with self.lock:
            return self.block_to_datanodes.get(block_id, [])


# ==============================================================================
# 3. MINI HDFS CLIENT API - Kullanıcının kullandığı üst düzey arayüz
# ==============================================================================
class MiniHDFS:
    """
    MiniHDFS: Kullanıcıya sunulan basit API.
    put(filename, data) -> dosya yaz
    get(filename)      -> dosya oku
    ls()               -> dosya listele
    """
    def __init__(self, replication_factor: int = 2, block_size: int = 1024):
        self.namenode = NameNode(replication_factor, block_size)

    def add_datanode(self, node_id: str = None) -> DataNode:
        """Yeni bir DataNode ekler."""
        if node_id is None:
            node_id = f"dn-{uuid.uuid4().hex[:6]}"
        dn = DataNode(node_id, f"localhost:{9000 + len(self.namenode.datanodes)}")
        self.namenode.register_datanode(dn)
        return dn

    def put(self, filename: str, data: bytes) -> bool:
        """Dosyayı HDFS'e yazar."""
        return self.namenode.store_file(filename, data)

    def get(self, filename: str) -> Optional[bytes]:
        """Dosyayı HDFS'den okur."""
        return self.namenode.retrieve_file(filename)

    def ls(self) -> List[str]:
        """Dosyaları listeler."""
        return self.namenode.list_files()

    def status(self):
        """Sistem durumunu gösterir."""
        print(f"\n{'='*60}")
        print(f"MINI HDFS STATUS")
        print(f"{'='*60}")
        print(f"DataNode sayısı: {len(self.namenode.datanodes)}")
        for dn in self.namenode.datanodes.values():
            print(f"  {dn}")
        print(f"Dosya sayısı: {len(self.namenode.file_to_blocks)}")
        for fname, blocks in self.namenode.file_to_blocks.items():
            print(f"  {fname}: {len(blocks)} blocks -> {blocks}")
        print(f"{'='*60}\n")


# ==============================================================================
# 4. TEST / DEMO FONKSİYONU
# ==============================================================================
def test_mini_hdfs():
    """
    Mini HDFS testi:
    1. 3 DataNode ekle
    2. Bir dosya yaz (replication=2 ile)
    3. Dosyayı oku
    4. Bir DataNode'u "düşür" ve hala okuyabildiğini göster (fault tolerance)
    5. DataNode'ları listele
    """
    print("\n" + "="*60)
    print("MINI HDFS DEMO")
    print("="*60)

    # HDFS sistemi oluştur (replication=2, block=50 bytes)
    hdfs = MiniHDFS(replication_factor=2, block_size=50)

    # DataNode'ları ekle
    dn1 = hdfs.add_datanode("sunucu-1")
    dn2 = hdfs.add_datanode("sunucu-2")
    dn3 = hdfs.add_datanode("sunucu-3")

    # Dosya yaz
    data = b"Merhaba bu mini HDFS test dosyasidir. Bu dosya blocklara bolunup birden fazla sunucuda saklanacaktir."
    success = hdfs.put("/user/test/dosya.txt", data)
    print(f"\nYazma basarili: {success}")

    # Dosyayı oku
    result = hdfs.get("/user/test/dosya.txt")
    print(f"\nOkunan veri: {result}")
    print(f"Orijinal ile eslesiyor: {result == data}")

    # Fault tolerance testi: bir DataNode'u düşür
    print(f"\n--- FAULT TOLERANCE TESTI ---")
    print(f"Sunucu-2 düsürülüyor...")
    hdfs.namenode.datanodes["sunucu-2"].alive = False

    # Hala okuyabilmeli (replication sayesinde)
    result2 = hdfs.get("/user/test/dosya.txt")
    print(f"Fault tolerance calisiyor: {result2 == data}")

    # Sistem durumu
    hdfs.status()

    return hdfs


if __name__ == "__main__":
    test_mini_hdfs()
