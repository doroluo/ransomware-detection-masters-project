"""Download and verify the frozen demo checkpoints; does not load model objects."""
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import stat
import urllib.request
import zipfile

from .predict import HERE,DEFAULT_BUNDLE,verify_bundle


def extract_verified(blob,destination,expected_sha256):
    if hashlib.sha256(blob).hexdigest()!=expected_sha256:raise ValueError('Downloaded checkpoint archive failed SHA-256 verification.')
    destination=Path(destination)
    if destination.exists() and any(destination.iterdir()):raise ValueError('Choose a new or empty checkpoint output folder.')
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive entries.')
        for entry in archive.infolist():
            path=PurePosixPath(entry.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in entry.filename or ':' in entry.filename:
                raise ValueError('Unsafe checkpoint archive path.')
            if stat.S_ISLNK(entry.external_attr>>16) or path.suffix not in ('.pt','.json'):
                raise ValueError('Unexpected checkpoint archive entry.')
        manifest=json.loads(archive.read('bundle_manifest.json'))
        if set(names)!=set(manifest['files_sha256'])|{'bundle_manifest.json'}:
            raise ValueError('Archive content differs from its manifest.')
        for name,digest in manifest['files_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError('Checkpoint member hash mismatch: '+name)
        destination.mkdir(parents=True,exist_ok=True)
        for name in names:
            target=destination/Path(name);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.read(name))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--archive',type=Path,help='Use the matching ZIP downloaded manually, with the same hash checks')
    args=parser.parse_args(argv)
    release=json.loads((HERE/'release.json').read_text())
    try:
        if args.out.exists() and any(args.out.iterdir()):
            verify_bundle(args.out)
            print('Existing checkpoint bundle verified: '+str(args.out));return 0
        if args.archive:blob=args.archive.read_bytes()
        else:
            print(f"Downloading {release['filename']} ({release['bytes']/1e6:.1f} MB)",flush=True)
            request=urllib.request.Request(release['url'],headers={'User-Agent':'RDMP-Demo-Checkpoints/1'})
            with urllib.request.urlopen(request,timeout=120) as response:blob=response.read(release['bytes']+1)
        if len(blob)!=release['bytes']:raise ValueError('Checkpoint archive has an unexpected size.')
        extract_verified(blob,args.out,release['sha256'])
        verify_bundle(args.out)
        print('Checkpoints downloaded and verified: '+str(args.out));return 0
    except (ValueError,FileNotFoundError,urllib.error.URLError) as error:parser.error(str(error))


if __name__=='__main__':raise SystemExit(main())
