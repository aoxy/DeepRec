import collections

import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm, trange

class Node:
    def __init__(self, key, val, pre=None, nex=None, freq=0):
        self.pre = pre
        self.nex = nex
        self.freq = freq
        self.val = val
        self.key = key
        
    def insert(self, nex):
        nex.pre = self
        nex.nex = self.nex
        self.nex.pre = nex
        self.nex = nex
    
def create_linked_list():
    head = Node(0, 0)
    tail = Node(0, 0)
    head.nex = tail
    tail.pre = head
    return (head, tail)

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.minFreq = 0
        self.freqMap = collections.defaultdict(create_linked_list)
        self.keyMap = {}

    def delete(self, node):
        if node.pre:
            node.pre.nex = node.nex
            node.nex.pre = node.pre
            if node.pre is self.freqMap[node.freq][0] and node.nex is self.freqMap[node.freq][-1]:
                self.freqMap.pop(node.freq)
        return node.key
        
    def increase(self, node):
        node.freq += 1
        self.delete(node)
        self.freqMap[node.freq][-1].pre.insert(node)
        if node.freq == 1:
            self.minFreq = 1
        elif self.minFreq == node.freq - 1:
            head, tail = self.freqMap[node.freq - 1]
            if head.nex is tail:
                self.minFreq = node.freq

    def get(self, key: int) -> int:
        if key in self.keyMap:
            self.increase(self.keyMap[key])
            return self.keyMap[key].val
        return None

    def put(self, key: int, value: int) -> None:
        if self.capacity != 0:
            if key in self.keyMap:
                node = self.keyMap[key]
                node.val = value
            else:
                node = Node(key, value)
                self.keyMap[key] = node
                self.size += 1
            self.increase(node)
            if self.size > self.capacity:
                self.size -= 1
                deleted = self.delete(self.freqMap[self.minFreq][0].nex)
                self.keyMap.pop(deleted)
                return deleted
            return None


class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:

    def __init__(self, capacity: int):
        self.cache = dict()
        # 使用伪头部和伪尾部节点    
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def get(self, key: int) -> int:
        if key not in self.cache:
            return None
        # 如果 key 存在，先通过哈希表定位，再移到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> int:
        if key not in self.cache:
            # 如果 key 不存在，创建一个新的节点
            node = DLinkedNode(key, value)
            # 添加进哈希表
            self.cache[key] = node
            # 添加至双向链表的头部
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                # 如果超出容量，删除双向链表的尾部节点
                removed = self.removeTail()
                # 删除哈希表中对应的项
                self.cache.pop(removed.key)
                self.size -= 1
                return removed.key
        else:
            # 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
            return None
    
    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node

def used_before_evict(visit_seq, CACHE, capacity, emb_size):
    load_count = 0
    save_count = 0
    pa_count = 4096 // emb_size
    extra_load = 0
    nouse_load = 0
    cache:LRUCache|LFUCache = CACHE(capacity)
    append_list = []
    keyindex = dict()
    extra_key = dict()
    for v in tqdm(visit_seq):
        emb = cache.get(v)
        # if v in extra_key:
        #     extra_key[v] = True
        if emb is None:
            inserts = [v]
            if v in keyindex:
                load_count += 1
                start = keyindex[v]
                end = min(len(append_list), start + pa_count)
                inserts = append_list[start:end]
            # siz1 = len(extra_key)
            for i in inserts:
                # extra_load += len(set(inserts)) - 1
                
                # if i != v:
                #     extra_key[i] = False
                
                ev = cache.put(i, i)
                if ev is not None:
                    keyindex[ev] = len(append_list)
                    save_count += 1
                    append_list.append(ev)
                    # if ev in extra_key and not extra_key[ev]:
                    #     nouse_load += 1
    return load_count, save_count, extra_load, nouse_load



EMBEDDING_COLS = pickle.load(open("EMBEDDING_COLS.pk", "rb"))
visit_seq_list = pickle.load(open("visit_seq.pk", "rb"))



MAX_LENGTH = 1_000_000

for i, visit_seq in enumerate(visit_seq_list):
    if EMBEDDING_COLS[i] in {"district_id", "times"}:
        continue
    print(EMBEDDING_COLS[i], len(visit_seq), len(set(visit_seq)))
    for emb_size in [256, 512, 1024, 2048, 4096]:
        load_count, save_count, extra_load, nouse_load= used_before_evict(visit_seq, LFUCache, int(2557 * 0.5), emb_size)
        print("emb_size: ", emb_size, "  ===== ",load_count, save_count, extra_load, nouse_load)
