import unittest
from types import SimpleNamespace
from tools.mysql_resize import wait_buffer_pool_ready


class ResizeTests(unittest.TestCase):
    def db(self, states):
        pending = list(states)
        def status(names):
            value = pending[0]
            if len(pending) > 1:
                pending.pop(0)
            return value
        return SimpleNamespace(
            show_variables=lambda names: {'innodb_buffer_pool_size': 2 * 1024**3, 'innodb_page_size': 16384},
            show_status=status,
        )

    def test_idle_code_alone_is_not_sufficient_and_two_good_polls_required(self):
        target_pages = 2 * 1024**3 // 16384
        states = [
            {'Innodb_buffer_pool_resize_status_code': 0, 'Innodb_buffer_pool_pages_total': target_pages // 2},
            {'Innodb_buffer_pool_resize_status_code': 2, 'Innodb_buffer_pool_pages_total': target_pages},
            {'Innodb_buffer_pool_resize_status_code': 0, 'Innodb_buffer_pool_pages_total': target_pages},
            {'Innodb_buffer_pool_resize_status_code': 0, 'Innodb_buffer_pool_pages_total': target_pages},
        ]
        now = [0.0]
        result = wait_buffer_pool_ready(self.db(states), 2 * 1024**3, timeout=2,
                                        clock=lambda: now[0], sleep=lambda n: now.__setitem__(0, now[0] + n))
        self.assertEqual(result['polls'], 4)
        self.assertEqual(result['page_accounted_bytes'], result['expected_bytes'])

    def test_failure_and_missing_status_fail_closed(self):
        for status, error in [({'Innodb_buffer_pool_resize_status_code': 7, 'Innodb_buffer_pool_pages_total': 131072}, RuntimeError),
                              ({}, ValueError)]:
            with self.assertRaises(error):
                wait_buffer_pool_ready(self.db([status]), 2 * 1024**3, timeout=1)

    def test_busy_resize_times_out(self):
        now = [0.0]
        db = self.db([{'Innodb_buffer_pool_resize_status_code': 2, 'Innodb_buffer_pool_pages_total': 131072}])
        with self.assertRaises(TimeoutError):
            wait_buffer_pool_ready(db, 2 * 1024**3, timeout=.3,
                                   clock=lambda: now[0], sleep=lambda n: now.__setitem__(0, now[0] + n))


if __name__ == '__main__':
    unittest.main()
