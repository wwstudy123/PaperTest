"""MySQL 8.0.31+ 缓冲池后台调整的有界完成检查；只读取状态。"""
from __future__ import annotations

import math
import time


def wait_buffer_pool_ready(db, expected_bytes, *, timeout=60, clock=time.monotonic, sleep=time.sleep):
    if (type(expected_bytes) is not int or expected_bytes <= 0
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('Positive buffer size and finite timeout required')
    started = clock()
    consecutive = 0
    polls = 0
    last = {}
    while clock() - started <= timeout:
        variables = db.show_variables(['innodb_buffer_pool_size', 'innodb_page_size'])
        status = db.show_status(['Innodb_buffer_pool_resize_status_code', 'Innodb_buffer_pool_pages_total'])
        try:
            code = int(status['Innodb_buffer_pool_resize_status_code'])
            pages = int(status['Innodb_buffer_pool_pages_total'])
            configured = int(variables['innodb_buffer_pool_size'])
            actual = pages * int(variables['innodb_page_size'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Cannot verify buffer-pool resize state; refusing measurement') from exc
        polls += 1
        last = {'status_code': code, 'configured_bytes': configured, 'page_accounted_bytes': actual,
                'expected_bytes': expected_bytes, 'polls': polls, 'wait_seconds': clock() - started}
        if code == 7:
            raise RuntimeError('MySQL reports buffer-pool resize failed')
        # 页计数仅作辅助一致性检查，允许内部记账开销；两次空闲读回避免接受旧状态。
        ready = code == 0 and configured == expected_bytes and abs(actual - expected_bytes) <= max(16384, expected_bytes * .01)
        consecutive = consecutive + 1 if ready else 0
        if consecutive >= 2:
            return last
        sleep(.15)
    raise TimeoutError(f'Buffer-pool resize did not settle: {last}')
