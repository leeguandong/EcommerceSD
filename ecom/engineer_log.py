# -*- coding: utf-8 -*-
import sys
import json
import traceback as tb
import logging

log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t%(process)d\t%(thread)d\t%(module)s\t%(funcName)s:\t%(message)s"
logging.basicConfig(
    format=log_format,
    filename='ecommercesd.log',
    level=logging.INFO)

log_pattern = '{{"code":{};"message":{} }}'


class Engineer(Exception):
    def __init__(self, err):
        super().__init__(err)


# 数值异常
class EngArithmetic(Engineer):
    def __init__(self, err):
        super().__init__(err)


# 算法异常
class EngAlgorithm(Engineer):
    def __init__(self, err):
        super().__init__(err)


# 函数参数异常
class EngParameter(Engineer):
    def __init__(self, err):
        super().__init__(err)


# IO异常
class EngIO(Engineer):
    def __init__(self, err):
        super().__init__(err)


# 网络异常
class EngNetwork(EngIO):
    def __init__(self, err):
        super().__init__(err)


class EngineerLog(object):
    def __init__(self):
        self.logModelx = logging

    def return_pattern(self, code=0, message='', result=''):
        # code:返回的代码
        # message:关于代码的英文解释
        # result: 返回的结果，必须是可被python序列化的
        pattern_di = {"code": 0,
                      "message": "",
                      "result": ""
                      }
        pattern_di["code"] = code
        pattern_di["message"] = message
        pattern_di['result'] = result

        # 添加json序列化方式，提前发现问题，防止返回值在uwsgi(flask)层暴露
        res = json.dumps(pattern_di)
        return pattern_di

    def _eng_log(self, level, meg):
        if not isinstance(meg, str):
            # meg = '输入错误内容，需要输入字符串，且是str类型'
            meg = "Enter the error content, which needs to be a string of str type"

        moduleName = sys._getframe().f_back.f_back.f_code.co_filename
        funcName = sys._getframe().f_back.f_back.f_code.co_name
        lineno = sys._getframe().f_back.f_back.f_lineno
        # string = "模块:{}; 函数:{}; 行:{}; 内容:{}"
        string = "module:{}; function:{}; line:{}; meg:{}"

        if level == 'info':
            self.logModelx.info(string.format(moduleName, funcName, lineno, meg))
        elif level == 'warn':
            self.logModelx.warning(string.format(moduleName, funcName, lineno, meg))
        elif level == 'error':
            self.logModelx.error(string.format(moduleName, funcName, lineno, meg))
        elif level == 'debug':
            self.logModelx.debug(string.format(moduleName, funcName, lineno, meg))
        else:
            raise

    def info(self, meg):
        self._eng_log('info', meg)

    def warn(self, meg):
        self._eng_log('warn', meg)

    def error(self, meg):
        self._eng_log('error', meg)

    def debug(self, meg):
        self._eng_log('debug', meg)


# --------------------
log = EngineerLog()
return_pattern = log.return_pattern


def test():
    try:
        1 / 0
    except:
        tb_str = tb.format_exc()
        log.info(tb_str, 'ValueError')


if __name__ == '__main__':
    test()
