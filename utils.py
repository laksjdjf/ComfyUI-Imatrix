import os
import struct
from typing import Dict, List, Tuple, Sequence, Mapping, Union

def load_imatrix(
    imatrix_file: str,
    trace_env: str = "LLAMA_TRACE",
) -> Tuple[Dict[str, List[float]], str, int]:
    """
    Parameters
    ----------
    imatrix_file : str
        読み込む .imatrix バイナリへのパス
    trace_env : str, default "LLAMA_TRACE"
        この環境変数がセットされていればデバッグ出力を行う

    Returns
    -------
    imatrix_data : dict[str, list[float]]
        エントリ名 → 重要度ベクトル
    imatrix_dataset : str
        ファイル末尾に埋め込まれたデータセット名（なければ ""）
    m_last_call : int
        行列計算時のチャンク数（なければ 0）
    """
    imatrix_data: Dict[str, List[float]] = {}
    imatrix_dataset = ""
    m_last_call = 0

    with open(imatrix_file, "rb") as f:
        # --- 1. 先頭: エントリ総数 ------------------------------------------
        n_entries_bytes = f.read(4)
        if len(n_entries_bytes) < 4:
            raise ValueError(f"{imatrix_file}: no data")
        n_entries = struct.unpack("<i", n_entries_bytes)[0]
        if n_entries < 1:
            raise ValueError(f"{imatrix_file}: n_entries < 1")

        # --- 2. 各エントリを読み取る ----------------------------------------
        for i in range(n_entries):
            # 2-a. 名前長と名前文字列
            name_len = struct.unpack("<i", f.read(4))[0]
            name = f.read(name_len).decode("utf-8", errors="replace")

            # 2-b. 呼び出し回数 ncall と 値の個数 nval
            ncall, nval = struct.unpack("<ii", f.read(8))
            if nval < 1:
                raise ValueError(f"entry {i}: nval < 1")

            # 2-c. nval 個の float32
            buf = f.read(4 * nval)
            if len(buf) < 4 * nval:
                raise ValueError(f"entry {i}: data truncated")
            values = list(struct.unpack(f"<{nval}f", buf))

            # 2-d. 平均化
            if ncall > 0:
                values = [v / ncall for v in values]

            imatrix_data[name] = values

            if os.getenv(trace_env):
                print(
                    f"load_imatrix: loaded data (size = {nval:6d}, "
                    f"ncall = {ncall:6d}) for '{name}'"
                )

        # --- 3. 末尾に追加メタ情報があるか確認 ------------------------------
        tail = f.read(4)
        if tail:  # まだバイトが残っている場合のみ
            m_last_call = struct.unpack("<i", tail)[0]
            dataset_len = struct.unpack("<i", f.read(4))[0]
            imatrix_dataset = f.read(dataset_len).decode("utf-8", errors="replace")
            if os.getenv(trace_env):
                print(f"load_imatrix: imatrix dataset = '{imatrix_dataset}'")

    print(
        f"load_imatrix: loaded {len(imatrix_data)} importance matrix entries "
        f"from {imatrix_file} computed on {m_last_call} chunks"
    )
    return imatrix_data, imatrix_dataset, m_last_call

def save_imatrix(
    imatrix_file: str,
    imatrix_data: Mapping[str, Sequence[float]],
    *,
    call_counts: Union[int, Mapping[str, int]] = 1,
    imatrix_dataset: str = "",
    m_last_call: int = 0,
) -> None:
    """
    Parameters
    ----------
    imatrix_file : str
        出力する .imatrix バイナリのパス
    imatrix_data : dict[str, Sequence[float]]
        エントリ名 → 値のベクトル（平均済み／未平均どちらでも OK）
    call_counts : int | dict[str, int], default 1
        ・各エントリ共通の呼び出し回数 (= 平均化係数) を 1 つの int で指定  
        ・あるいはエントリごとに dict で指定  
        ※「すでに平均済みの値」を保存したいときは 0 を渡してください
    imatrix_dataset : str, default ""
        ファイル末尾に埋め込むデータセット名（空文字なら書き込まない）
    m_last_call : int, default 0
        全体のチャンク数。dataset 名を書くときはセットで入れる
    """
    # 入力バリデーション ---------------------------------------------------
    if isinstance(call_counts, int):
        call_counts = {k: call_counts for k in imatrix_data.keys()}
    else:
        # dict で来た場合、全キーがそろっているか確認
        missing = set(imatrix_data) - set(call_counts)
        if missing:
            raise KeyError(f"call_counts is missing keys: {missing}")

    with open(imatrix_file, "wb") as f:
        # 1. 先頭: エントリ総数 -------------------------------------------
        n_entries = len(imatrix_data)
        f.write(struct.pack("<i", n_entries))

        # 2. 各エントリを順番に書き込む -----------------------------------
        for name, values in imatrix_data.items():
            name_b = name.encode("utf-8")
            f.write(struct.pack("<i", len(name_b)))   # name length
            f.write(name_b)                           # name bytes

            ncall = call_counts[name]
            nval = len(values)
            f.write(struct.pack("<ii", ncall, nval))  # ncall, nval

            # nval 個の float32
            fmt = f"<{nval}f"
            f.write(struct.pack(fmt, *values))

        # 3. オプションのメタ情報 -----------------------------------------
        if imatrix_dataset:
            f.write(struct.pack("<i", m_last_call))
            dataset_b = imatrix_dataset.encode("utf-8")
            f.write(struct.pack("<i", len(dataset_b)))
            f.write(dataset_b)
