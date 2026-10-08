"""Rebuild exact paper MOTChallenge GT from public test annotations and protocol identities."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import stat
import zipfile

try:
    from .score_mot import GT_MEMBER_PATTERN, content_set_sha256
except ImportError:
    from score_mot import GT_MEMBER_PATTERN, content_set_sha256

IDENTITY_SHA256 = "c6e213fd3686d876c26889cb7fdcc6ebcc37c5b629d812cc163b0cdd0a28fc89"


def render_members(data, identity_rows):
    identities = {(video, track): identity for video, track, identity in identity_rows}
    if len(identities) != len(identity_rows):
        raise ValueError("Duplicate protocol identity key")
    videos = {video['id']: video['name'] for video in data['videos']}
    images = {image['id']: image for image in data['images']}
    rows = defaultdict(list)
    for annotation in data['annotations']:
        if annotation['category_id'] == 4:
            continue
        image = images[annotation['image_id']]
        video = image['video_id']
        if annotation['video_id'] != video:
            raise ValueError("Annotation/image video mismatch")
        track = annotation.get('paper_track_id', annotation['track_id'])
        identity = identities[(video, track)]
        rows[video].append((int(image['frame_index']) + 1, int(identity), *annotation['bbox']))
    files = {}
    duplicates = total = 0
    for video, values in rows.items():
        name = videos[video] + '.txt'
        if not GT_MEMBER_PATTERN.fullmatch(name) or name in files:
            raise ValueError("Unsafe or duplicate GT basename")
        values.sort(key=lambda row: (row[0], row[1]))
        files[name] = ''.join('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n' % row for row in values).encode()
        duplicates += sum(count - 1 for count in Counter((row[0], row[1]) for row in values).values())
        total += len(values)
    return files, total, duplicates


def encode_zip(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw, compresslevel=9)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('Output must be a new file')
    mapping_bytes = Path(__file__).with_name('paper_gt_identities.json').read_bytes()
    if hashlib.sha256(mapping_bytes).hexdigest() != IDENTITY_SHA256:
        raise ValueError('Protocol identity mapping SHA mismatch')
    protocol = json.loads(mapping_bytes)
    source = args.annotation.read_bytes()
    files, rows, duplicates = render_members(json.loads(source), protocol['identities'])
    content_digest = content_set_sha256(files)
    if (len(files), rows, duplicates) != (64, 337009, 423) or content_digest != protocol['gt_content_set_sha256']:
        raise ValueError('Annotations do not reconstruct exact paper GT: content SHA/count mismatch')
    raw = encode_zip(files)
    archive_digest = hashlib.sha256(raw).hexdigest()
    if archive_digest != protocol['gt_zip_sha256']:
        raise ValueError('Archive SHA mismatch; check the local zlib version')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        handle.write(raw)
    print(json.dumps({'annotation_sha256': hashlib.sha256(source).hexdigest(),
                      'gt_zip_sha256': archive_digest, 'content_set_sha256': content_digest,
                      'members': len(files), 'rows': rows, 'duplicate_frame_id_rows': duplicates}))


if __name__ == '__main__':
    main()
