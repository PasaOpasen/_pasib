
from pathlib import Path

from collections import Counter
import pickle


def compress(text: str, iterations: int = 10000):

    table = []
    text = list(text)

    for k in range(1, iterations + 1):
        if len(text) < 1:
            break

        (v1, v2), count = Counter(zip(text[:-1], text[1:])).most_common(1)[0]
        if count < 2:
            break
        table.append((k, (v1, v2)))

        t = []
        last_ok_i = -1
        last_index = len(text) - 1

        i = 0
        while i < last_index:
            s = text[i]
            if s == v1 and text[i + 1] == v2:
                t.append(k)
                last_ok_i = i + 1
                i += 2
            else:
                t.append(s)
                i += 1

        if last_ok_i != last_index:  # last is not appended
            t.append(text[-1])

        text = t

    print(f"iterations = {k + 1}")

    t = [text[0]]
    for v in text[1:]:
        if isinstance(v, str) and isinstance(t[-1], str):
            t[-1] += v
        else:
            t.append(v)
    text = t

    return table, text


def int_to_bytes(v: int):
    import math
    return v.to_bytes(math.ceil((v + 1) / 256), 'little')


if __name__ == '__main__':
    tb, tt = compress(
        # 'curl --proxy http://localhost:8051 https://www.baidu.com --include --verbose'
        Path('paths.yaml').read_text(encoding='utf-8')
    )

    import numpy as np

    np.savez(
        'res.npz',
        arr=np.array(
            #sum(([k, v1, v2] for k, (v1, v2) in tb), []) + [0] + tt,
            tt,
            dtype=object
        )
    )

    print()
    # txt = '&'.join(
    #     f"{k}${('' if isinstance(v1, int) else '.') + str(v1)}${('' if isinstance(v2, int) else '.') + str(v2)}"
    #     for k, (v1, v2) in tb
    # ) + '&' + '&'.join(
    #     ('|' if isinstance(s, str) else '') + str(s)
    #     for s in tt
    # )

    # br = bytearray()
    # for k, (v1, v2) in tb:
    #     br.append(int_to_bytes(k))

    # s = 0
    # for k, (v1, v2) in tb:
    #     s += 1 + len(int_to_bytes(k))
    #     for h in (v1, v2):
    #         if isinstance(h, str):
    #             s += 1 + len(h.encode('utf8'))
    #         else:
    #             s += 1 + len(int_to_bytes(h))
    # print(s)
    #
    # s += 2
    # for v in tt:
    #     if isinstance(v, str):
    #         s += 1 + len(v.encode('utf8'))
    #     else:
    #         s += 1 + len(int_to_bytes(v))
    #
    # print(s / 1024)
    # print()

    # Path('result.yaml').write_text(txt, encoding='utf-8')
    # with open('result.pkl', 'wb') as f:
    #     pickle.dump((tb, tt), f)

