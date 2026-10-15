"""Render synthetic typeset answer stages. No student handwriting or private inputs."""
import argparse
import json
from pathlib import Path


STAGES = [
    ['Take right as positive.', 'Net force = 10 - 2 = 8 N (right)'],
    ['Take right as positive.', 'Net force = 10 - 2 = 8 N (right)', 'a = 8 / 2 = 4 m/s^2 (right)'],
    ['Take right as positive.', 'Net force = 10 - 2 = 8 N (right)', 'a = 8 / 2 = 4 m/s^2 (right)', 'v = 0 + 4 x 3 = 12 m/s (right)'],
]


def generate(output, video=False):
    from PIL import Image, ImageDraw, ImageFont
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = [f'force_stage_{i}.png' for i in range(1, 4)] + ['force_error.png']
    if video:
        names += ['force_process.mp4']
    if any((output / n).exists() for n in names + ['media_manifest.json']):
        raise ValueError('Choose an empty output location; existing media will not be overwritten')
    frames = []
    for i, lines in enumerate(STAGES + [['Net force = 10 + 2 = 12 N', 'a = 12 / 2 = 6 m/s^2']]):
        im = Image.new('RGB', (960, 640), 'white')
        draw = ImageDraw.Draw(im)
        draw.text((35, 30), 'AI-AUTHORED TYPESET DEMO / public_force_01', fill='navy', font=ImageFont.load_default(size=25))
        for y in range(130, 610, 65):
            draw.line((30, y, 930, y), fill='#e2eaf5')
        for j, line in enumerate(lines):
            draw.text((40, 155 + j * 80), line, fill='black', font=ImageFont.load_default(size=30))
        im.save(output / names[i])
        if i < 3:
            frames.append(im)
    if video:
        import cv2
        import numpy as np
        writer = cv2.VideoWriter(str(output / 'force_process.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (960, 640))
        if not writer.isOpened():
            raise RuntimeError('MP4 encoder unavailable; photo files remain usable')
        try:
            for frame in frames:
                for _ in range(40):
                    writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
    result = {'source': 'AI-authored typeset engineering input; not real student handwriting',
              'problem_id': 'public_force_01', 'stages': STAGES, 'files': names,
              'error_example': 'The last image deliberately adds the opposing force.',
              'video': '24s, three 8s holds; holds do not represent cognitive stalls' if video else 'not requested'}
    (output / 'media_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'output/public_demo_media')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.video), indent=2))
