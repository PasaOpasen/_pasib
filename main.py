

from typing import Tuple, List, Union
from typing_extensions import TypeAlias

from pathlib import Path

from collections import Counter

import numpy as np


array: TypeAlias = np.ndarray

array1Du8: TypeAlias = array
array1Du16: TypeAlias = array

array1Di8: TypeAlias = array
array1Di16: TypeAlias = array

array2Du8: TypeAlias = array
array2Du16: TypeAlias = array

array2Di8: TypeAlias = array
array2Di16: TypeAlias = array


def arrays_unsqueeze(ars: List[Union[array1Di16, Tuple[array1Di16, array1Du8]]]) -> List[Union[array1Di16, array1Du8]]:
    res = []
    for r in ars:
        if isinstance(r, tuple):
            res.extend(r)
        else:
            res.append(r)
    return res


def arr_down_size(arr: array1Di16) -> List[Union[array1Di16, Tuple[array1Di16, array1Du8]]]:

    result = []

    i = 0
    max_index = arr.size - 1

    while True:
        s = set()
        k = i

        while k <= max_index and len(s) <= 255:
            s.add(arr[k])
            k += 1

            # if len(s) > 30 and len(s) > (k - i) // 2 - 2:  # no sense to continue ?
            #     break

        if s:

            if len(s) < (k - i) / 2:
                values = np.array(sorted(s), dtype=np.int16)
                i16_to_u8 = {v: i for i, v in enumerate(values)}
                _arr = np.array([i16_to_u8[v] for v in arr[i: k]], dtype=np.uint8)

                result.append((values, _arr))
            else:
                result.append(arr[i: k].astype(np.int16))

            i = k
        else:
            break

    return result



def text_to_table_and_map(text: str, iterations: int = 10000) -> Tuple[str, array1Di16, array1Di16]:

    symbols = ''.join(k for k, _ in Counter(text).most_common())

    s_map = {k: i for i, k in enumerate(symbols, 1)}

    arr = np.array(
        [s_map[s] for s in text]
    )

    n_map = []
    s_count = len(symbols) + 1

    for k in range(s_count, s_count + iterations):
        if arr.size < 2:
            break

        (v1, v2), count = Counter(zip(arr[:-1], arr[1:])).most_common(1)[0]
        if count < 4:
            break

        n_map.extend((-k, v1, v2) if v1 != v2 else (k, v1))

        _arr = np.empty_like(arr)
        last_ok_i = -1
        last_index = arr.size - 1

        i = j = 0
        while i < last_index:
            s = arr[i]

            if s == v1 and arr[i + 1] == v2:
                _arr[j] = k
                last_ok_i = i + 1
                i += 2
            else:
                _arr[j] = s
                i += 1

            j += 1

        if last_ok_i != last_index:  # last is not appended
            _arr[j] = arr[last_index]

        arr = _arr[:j+1]

    print(f"iterations = {k - s_count}")

    return symbols, np.array(n_map).astype(np.int16), arr.astype(np.int16)


def int_to_bytes(v: int):
    import math
    return v.to_bytes(math.ceil((v + 1) / 256), 'little')


if __name__ == '__main__':
    ss, nm, rr = text_to_table_and_map(
        # 'curl --proxy http://localhost:8051 https://www.baidu.com --include --verbose'
        Path('paths.yaml').read_text(encoding='utf-8')
    )

    np.savez_compressed(
        'res.npz',
        symbols=np.array([ss]),
        maps=nm,
        text=rr
    )

    print(Path('res.npz').stat().st_size)


    nm_z = arr_down_size(nm)
    rr_z = arr_down_size(rr)

    np.savez_compressed(
        'res2.npz',
        *arrays_unsqueeze(nm_z), *arrays_unsqueeze(rr_z),
        symbols=np.array([ss]),
    )
    print(Path('res2.npz').stat().st_size)




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

