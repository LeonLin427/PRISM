"""Extract APK static features into features/{App}_v{version}.json.

Parse binary Manifest AXML with Androguard for package/version information,
permissions, metadata keys, and components. Scan DEX class descriptors and
aggregate two- and three-level package prefixes with counts. Sensitive API
features count method-name substrings in raw DEX bytes, not call sites or runtime
invocations. Also record native libraries, asset names, and domains.

Existing outputs with the same schema and APK modification time are skipped."""
import hashlib
import json
import os
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

# Silence Androguard/loguru, including when spawned workers import this module.
try:
    from loguru import logger
    logger.remove()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('SDK_PROJECT_ROOT', os.path.dirname(HERE))
APKS = os.path.join(ROOT, 'apks')
FEAT_DIR = os.path.join(HERE, 'features')

CLASS_RE = re.compile(rb'L([a-zA-Z][\w$]{0,60}(?:/[\w$]{1,60}){1,8});')
URL_RE = re.compile(rb'https?://([a-zA-Z0-9](?:[a-zA-Z0-9.-]{0,120})?\.[a-zA-Z]{2,20})')

# Increment when extraction semantics change so stale cached outputs are rebuilt.
SCHEMA_V = 2

# Sensitive method-name substrings counted in DEX bytes, not runtime calls.
# Notice-relevant behaviors: identifiers, app lists, location, clipboard, JS bridges, dynamic loading.
SENSITIVE_APIS = [
    b'getDeviceId',                # IMEI
    b'getImei',                    # IMEI (API 26+)
    b'getMeid',
    b'getSubscriberId',            # IMSI
    b'getSimSerialNumber',         # ICCID
    b'getLine1Number',             # Local phone number.
    b'getSerial',                  # Hardware serial number.
    b'getMacAddress',
    b'getCellLocation',
    b'getLastKnownLocation',
    b'getInstalledPackages',       # Installed application inventory.
    b'getInstalledApplications',
    b'getRunningTasks',            # Running tasks.
    b'getRunningAppProcesses',
    b'getPrimaryClip',             # Clipboard reads.
    b'setPrimaryClip',             # Clipboard writes.
    b'addJavascriptInterface',     # WebView JavaScript bridge, potentially relevant to remote code execution.
    b'DexClassLoader',             # Dynamic DEX loading, including updates or downloaded code.
]

# Exclude uninformative domain noise.
DOMAIN_NOISE = {
    'schemas.android.com', 'www.w3.org', 'w3.org', 'apache.org', 'www.apache.org',
    'opensource.org', 'gnu.org', 'www.gnu.org', 'json.org', 'ietf.org', 'www.ietf.org',
    'example.com', 'localhost', 'github.com', 'maven.apache.org', 'developer.android.com',
    'source.android.com', 'play.google.com', 'www.google.com', 'googleapis.com',
    'gstatic.com', 'www.gstatic.com', 'ssl.gstatic.com', 'w3c.org', 'www.w3c.org',
    'openjdk.java.net', 'java.sun.com', 'www.oracle.com', 'oracle.com', 'sun.com',
    'wx.qq.com', 'mp.weixin.qq.com', 'open.weixin.qq.com',  # Required by WeChat sharing; treated as noise.
}


def extract_manifest(apk_path):
    try:
        from androguard.core.apk import APK
        a = APK(apk_path)
        meta = []
        try:
            meta = a.get_elements('meta-data', 'name') or []
        except Exception:
            pass
        return {
            'package': a.get_package(),
            'version_name': a.get_androidversion_name(),
            'version_code': str(a.get_androidversion_code()),
            'permissions': sorted(a.get_permissions() or []),
            'meta_keys': sorted(set(meta)),
            'activities': a.get_activities() or [],
            'services': a.get_services() or [],
            'receivers': a.get_receivers() or [],
            'providers': a.get_providers() or [],
        }
    except Exception as e:
        return {'manifest_error': str(e)[:200]}


def extract_dex_features(zf):
    prefix2 = {}
    prefix3 = {}
    domains = set()
    api_calls = {}
    for name in zf.namelist():
        if not re.fullmatch(r'classes\d*\.dex', name):
            continue
        data = zf.read(name)
        for m in CLASS_RE.finditer(data):
            parts = m.group(1).decode('latin1').split('/')
            if len(parts) >= 2:
                p2 = '.'.join(parts[:2])
                prefix2[p2] = prefix2.get(p2, 0) + 1
            if len(parts) >= 3:
                p3 = '.'.join(parts[:3])
                prefix3[p3] = prefix3.get(p3, 0) + 1
        for m in URL_RE.finditer(data):
            d = m.group(1).decode('latin1').lower().strip('.')
            base = '.'.join(d.split('.')[-2:]) if d.count('.') >= 1 else d
            if d not in DOMAIN_NOISE and base not in DOMAIN_NOISE:
                domains.add(d)
        for api in SENSITIVE_APIS:
            api_calls[api.decode()] = api_calls.get(api.decode(), 0) + data.count(api)
    return prefix2, prefix3, sorted(domains), api_calls


def extract_one(apk_path, folder):
    feat = {'folder': folder, 'apk_file': os.path.basename(apk_path)}
    with open(apk_path, 'rb') as f:
        feat['md5'] = hashlib.md5(f.read()).hexdigest()
    feat['size'] = os.path.getsize(apk_path)

    with zipfile.ZipFile(apk_path) as zf:
        names = zf.namelist()
        feat['so_libs'] = sorted({
            os.path.basename(n) for n in names
            if n.startswith('lib/') and n.endswith('.so')
        })
        feat['assets'] = sorted({
            '/'.join(n.split('/')[:2]) for n in names if n.startswith('assets/')
        })[:500]
        p2, p3, domains, api_calls = extract_dex_features(zf)
        feat['prefix2'] = p2
        feat['prefix3'] = p3
        feat['domains'] = domains
        feat['api_calls'] = api_calls

    feat.update(extract_manifest(apk_path))
    feat['schema_v'] = SCHEMA_V
    return feat


def process_apk(task):
    apk_path, folder, out_path, mtime = task
    try:
        feat = extract_one(apk_path, folder)
        feat['apk_mtime'] = mtime
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(feat, f, ensure_ascii=False, indent=1)
        return (os.path.basename(apk_path), True, '')
    except Exception as e:
        return (os.path.basename(apk_path), False, str(e)[:200])


def main():
    os.makedirs(FEAT_DIR, exist_ok=True)
    tasks = []
    skipped = 0
    for folder in sorted(os.listdir(APKS)):
        subdir = os.path.join(APKS, folder)
        if not os.path.isdir(subdir):
            continue
        for f in sorted(os.listdir(subdir)):
            if not f.lower().endswith('.apk'):
                continue
            apk_path = os.path.join(subdir, f)
            out_path = os.path.join(FEAT_DIR, os.path.splitext(f)[0] + '.json')
            mtime = os.path.getmtime(apk_path)
            if os.path.exists(out_path):
                try:
                    with open(out_path, encoding='utf-8') as fp:
                        old = json.load(fp)
                    if old.get('apk_mtime') == mtime and old.get('schema_v') == SCHEMA_V:
                        skipped += 1
                        continue
                except Exception:
                    pass
            tasks.append((apk_path, folder, out_path, mtime))

    print(f'共 {len(tasks)} 个 APK 待提取, {skipped} 个已有缓存跳过')
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(process_apk, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            name, success, err = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                print(f'  [失败] {name}: {err}')
            if i % 20 == 0 or i == len(tasks):
                print(f'  进度 {i}/{len(tasks)}')
                sys.stdout.flush()
    print(f'\n完成: 成功 {ok}, 失败 {fail}, 缓存跳过 {skipped}')


if __name__ == '__main__':
    main()
