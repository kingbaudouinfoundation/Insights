"""
worker for enabling background workers to run
"""
import os
from threading import Thread

import click
from flask import Flask
from flask.cli import AppGroup

from rq import Worker, Queue, Connection
from rq.job import Job
from rq_win import WindowsWorker

from ..data.cache import redis_cache

cli = AppGroup('worker')

kill_key = "rq:jobs:kill"


# https://github.com/rq/rq/issues/684#issuecomment-352152369
class JobWithKill(Job):
    def kill(self):
        """ Force kills the current job causing it to fail """
        if self.is_started:
            self.connection.sadd(kill_key, self.get_id())

    def _execute(self):
        def check_kill(conn, id, interval=1):
            while True:
                res = conn.srem(kill_key, id)
                if res > 0:
                    os.kill(os.getpid(), signal.SIGKILL)
                time.sleep(interval)

        t = Thread(target=check_kill, args=(self.connection, self.get_id()))
        t.start()

        return super()._execute()


class QueueWithKill(Queue):
    job_class = JobWithKill


class WorkerWithKill(Worker):
    queue_class = QueueWithKill
    job_class = JobWithKill


class WindowsWorkerWithKill(WindowsWorker):
    queue_class = QueueWithKill
    job_class = JobWithKill

@cli.command('start')
def cli_start_worker():

    listen = ['high', 'default', 'low']

    conn = redis_cache()

    with Connection(conn):
        if hasattr(os, 'fork'):
            worker = WorkerWithKill(map(QueueWithKill, listen))
        else:
            worker = WindowsWorkerWithKill(map(QueueWithKill, listen))
        worker.work()
