#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    config.py
# @Author:      Kuro
# @Time:        1/18/2025 11:05 AM
import yaml


class Config:
    with open("./api/config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
