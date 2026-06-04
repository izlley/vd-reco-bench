"""Pod auto-shutdown 회피용 keep-alive.

본 pod 는 GPU 사용률이 20% 아래로 떨어지면 자동 종료된다. 이 스크립트
는 모든 GPU 에 가벼운 matmul 을 반복 실행하여 사용률을 유지한다.

사용법:
    python scripts/keep_alive.py &        # background 로
    kill <PID>                            # 중지
"""

from __future__ import annotations

import time

import torch


def main() -> None:
    n = torch.cuda.device_count()
    print(f"[keep_alive] {n} GPUs detected, starting...")
    if n == 0:
        return
    # 적당한 크기 (HBM 1-2GB 점유, util ~30%)
    tensors_a = [torch.randn(4096, 4096, device=f"cuda:{i}") for i in range(n)]
    tensors_b = [torch.randn(4096, 4096, device=f"cuda:{i}") for i in range(n)]
    try:
        while True:
            for i in range(n):
                # 약간의 matmul + sync 로 GPU 활용
                _ = tensors_a[i] @ tensors_b[i]
            torch.cuda.synchronize()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("[keep_alive] stopped.")


if __name__ == "__main__":
    main()
