"""
worker for enabling background workers to run
"""
import os

import click
from flask import Flask
from flask.cli import AppGroup, with_appcontext

from rq import Worker, Queue, Connection
from rq_win import WindowsWorker

from ..data.cache import get_cache

cli = AppGroup('worker')


@cli.command('start')
@with_appcontext
def cli_start_worker():

    listen = ['high', 'default', 'low']
    conn = get_cache()

    with Connection(conn):
        if hasattr(os, 'fork'):
            worker = Worker(map(Queue, listen))
        else:
            worker = WindowsWorker(map(Queue, listen))
        worker.work()
