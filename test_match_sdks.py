"""Regression tests for conservative SDK package matching."""

from match_sdks import match_one


def sig(package=None, **extra):
    item = {
        'name': 'target',
        'packages': [package] if package else [],
        'so_libs': [],
        'meta_keys': [],
        'domains': [],
        'assets': [],
    }
    item.update(extra)
    return item


def names(feat, signature):
    return {item['name'] for item, _rules in match_one(feat, [signature])}


def test_shallow_vendor_prefix_cannot_imply_deeper_sdk():
    feat = {'prefix2': {'com.tencent': 100}, 'prefix3': {'com.tencent.bugly': 10}}
    assert names(feat, sig('com.tencent.face')) == set()


def test_equal_or_deeper_observed_prefix_matches():
    assert names({'prefix3': {'com.tencent.face': 3}}, sig('com.tencent.face')) == {'target'}
    assert names({'prefix3': {'com.qiniu.android': 3}}, sig('com.qiniu')) == {'target'}


def test_manifest_component_can_support_deep_package():
    feat = {'activities': ['com.alibaba.mtl.appmonitor.AppMonitorActivity']}
    assert names(feat, sig('com.alibaba.mtl.appmonitor')) == {'target'}


def test_deep_package_without_specific_evidence_is_not_inferred():
    feat = {'prefix3': {'com.alibaba.mtl': 20}}
    assert names(feat, sig('com.alibaba.mtl.appmonitor')) == set()


def test_nonpackage_evidence_remains_available():
    feat = {
        'so_libs': ['libtarget.so'],
        'meta_keys': ['target.key'],
        'domains': ['api.example.com'],
        'assets': ['target.dat'],
    }
    signature = sig(None, so_libs=['libtarget.so'], meta_keys=['target.key'],
                    domains=['example.com'], assets=['target.dat'])
    hits = match_one(feat, [signature])
    assert names(feat, signature) == {'target'}
    assert len(hits[0][1]) == 4


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
    print(f'{len(tests)} matching regression tests passed')
