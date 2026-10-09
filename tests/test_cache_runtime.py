from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import threading
from pathlib import Path
import unittest
from convxai.cache import ByteLRU, CacheKey
from convxai.runtime import Audit, Gateway, Quota, RateLimited, ServiceUnavailable, Settings

class CacheTests(unittest.TestCase):
    def test_ttl_and_copy_isolation(self):
        now=[0];c=ByteLRU(ttl=5,clock=lambda:now[0])
        value,hit=c.get_or_load('a',lambda:[1]);value.append(2)
        self.assertEqual(c.get_or_load('a',lambda:[]),([1],True))
        now[0]=6
        self.assertEqual(c.get_or_load('a',lambda:[3]),([3],False))

    def test_lru_and_byte_bound(self):
        c=ByteLRU(max_entries=2,max_bytes=100)
        for k in ['a','b','a','c']: c.get_or_load(k,lambda:k)
        self.assertFalse(c.get_or_load('b',lambda:'b')[1])
        c.get_or_load('oversize',lambda:'x'*1000)
        self.assertLessEqual(c.bytes,100)
        self.assertNotIn('oversize',c._entries)

    def test_single_flight(self):
        c=ByteLRU();started=threading.Event();release=threading.Event();calls=[]
        def loader(): calls.append(1);started.set();release.wait(3);return [1]
        with ThreadPoolExecutor(max_workers=4) as pool:
            first=pool.submit(c.get_or_load,'key',loader);started.wait(3)
            others=[pool.submit(c.get_or_load,'key',loader) for _ in range(3)]
            release.set()
            self.assertEqual(first.result()[0],[1])
            self.assertTrue(all(f.result()[0]==[1] for f in others))
        self.assertEqual(len(calls),1)

    def test_failed_load_retry(self):
        c=ByteLRU()
        with self.assertRaises(ValueError): c.get_or_load('a',lambda: (_ for _ in ()).throw(ValueError()))
        self.assertEqual(c.get_or_load('a',lambda:2),(2,False))

    def test_invalidation_during_load(self):
        c=ByteLRU();started=threading.Event();release=threading.Event()
        def loader(): started.set();release.wait(3);return 1
        with ThreadPoolExecutor() as pool:
            f=pool.submit(c.get_or_load,'d/key',loader);started.wait(3)
            c.invalidate('d/');release.set();f.result()
        self.assertEqual(c.stats()['entries'],0)

    def test_key_versions(self):
        key=CacheKey('d','model','shap','local','i')
        self.assertNotEqual(key.digest,CacheKey('d','newmodel','shap','local','i').digest)
        self.assertNotEqual(key.digest,CacheKey('d','model','shap','local','i',language='fr').digest)

class RuntimeTests(unittest.TestCase):
    def test_quota_reset(self):
        now=[0];q=Quota(1,10,lambda:now[0]);q.reserve(5)
        with self.assertRaises(RateLimited): q.reserve(1)
        now[0]=61;q.reserve(1)

    def test_token_quota(self):
        with self.assertRaises(RateLimited): Quota(10,5).reserve(6)

    def test_bounded_backoff_and_circuit(self):
        waits=[];calls=[];now=[0]
        def fail(*args): calls.append(1);raise OSError()
        g=Gateway(transport=fail,sleep=waits.append,clock=lambda:now[0])
        with self.assertRaises(ServiceUnavailable): g.complete([{'content':'hello'}])
        self.assertEqual(len(calls),3);self.assertEqual(len(waits),2)
        with self.assertRaises(ServiceUnavailable): g.complete([{'content':'hello'}])
        self.assertEqual(len(calls),3)
        now[0]=31;g.transport=lambda *args:'ok'
        self.assertEqual(g.complete([{'content':'hello'}]),'ok')

    def test_metadata_only_audit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'audit.jsonl'
            Audit(p).record(intent='LOCALE',question='private text',response='secret',latency_ms=5)
            row=json.loads(p.read_text())
            self.assertNotIn('question',row);self.assertNotIn('response',row)
            self.assertEqual(row['intent'],'LOCALE')
