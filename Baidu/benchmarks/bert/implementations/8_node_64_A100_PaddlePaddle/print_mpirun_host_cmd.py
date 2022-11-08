# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

if __name__ == "__main__":
    env_name = "PADDLE_TRAINER_ENDPOINTS"
    env_value = os.environ.get(env_name)
    if env_value:
        ips = [s.split(':')[0] for s in env_value.split(',') if s.strip()]
    else:
        ips = []

    new_ips = []
    for ip in ips:
        new_ips.append(ip)
        new_ips.append(ip)
    ips = new_ips
    ips = ','.join(ips)
    if ips:
        print('--host {}'.format(ips))
