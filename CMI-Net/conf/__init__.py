""" dynamically load settings

author axiumao
"""
import conf.global_settings as settings

class Settings:
    def __init__(self, settings):

        for attr in dir(settings):
            if attr.isupper(): # 如果属性名是全大写
                setattr(self, attr, getattr(settings, attr)) # 设置属性值

settings = Settings(settings)
