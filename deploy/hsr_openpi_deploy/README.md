# hsr-openpi model with airoa-evaluation-vla

airoa-evalation-vlaのopenpi_policy_wrapperのポリシークラスにモデルの出力を送るためのlcmノード．

## 事前準備

1. airoa-evalation-vla/openpi_policy_wrapper/README.mdの環境構築の箇所を実行．
https://github.com/airoa-org/airoa-evaluation-vla/tree/main/openpi_policy_wrapper

2. airoa-evalation-vla/openpi_policy_wrapper/lcm_msgsをdocker内で使うので，docker-compose.yamlの書き換え
```bash
      - ../msgs/lcm_msgs:/root/catkin_ws/lcm_msgs
```
の行の，左のパートを上記の場所になるように変更

3. hsr-openpiの環境構築に従って準備


## 実行

dockerに入ってノードを実行すれば，lcmでpolicyとやり取り可能


hostで
```bash
./RUN-DOCKER-CONTAINER.sh
```

docker内で
```bash
export LCM_DEFAULT_URL=udpm://239.255.76.67:7667?ttl=1
cd /home/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 自分の環境では，これをpyproject.tomlに追加するとsync出来なくなったので，手動追加
uv pip install lcm

# 環境内に入って実行
source .venv/bin/activate
roscd
python hsr_openpi_lcm.py
```

"start server..."とプリントされたら，動く準備が出来ている

