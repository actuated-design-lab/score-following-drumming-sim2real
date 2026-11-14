# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""user0/, user1/, ... を自動で読み込み、各 user の gym.register を実行する。

新しい user フォルダ（user3/ など）を追加しても、このファイルを編集する必要はない。
読み込むのは名前が "user" で始まるサブパッケージだけ（common/ は各 user から import される）。
"""

import importlib
import pkgutil

_USER_PACKAGES = sorted(
    m.name for m in pkgutil.iter_modules(__path__) if m.ispkg and m.name.startswith("user")
)

for _name in _USER_PACKAGES:
    importlib.import_module(f".{_name}", __name__)
