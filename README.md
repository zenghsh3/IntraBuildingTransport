# ElevatorSimulator
> ElevatorSimulator是一电梯调度模拟环境

# 背景
电梯调度是模拟一栋大楼（Mansion）内，多部电梯(Elevator)相互协作运送乘客上下楼需求的一个模拟器。

# 入口
python demo.py config.ini
(需要python3.5或者更高版本)

# 主要结构
mansion/: 

  elevator.py:

    一部电梯的状态和控制过程, 定义在elevator.py文件内, 主要定义了Elevator类
    Elevator有自身完整的控制逻辑，包括，包括装载乘客，送达目的地，以及依照调度指令去到指定楼层，其核心的一些变量如下
      _target_floors ： 已经预定必须要经过的楼层，全部为装载车乘客所按的楼层
      _clicked_buttons:  用户按的目标电梯楼层
      _dispatch_target:  调度系统调度的目标楼层
      _dispatch_target_direction： 调度系统调度的目标方向，当电梯无方向到达调度楼层时，会将其方向改为调度的目标方向
      _load_person： 一个list的dict,每个floor对应的key存储将要去这个floor的所有person
      _exiting_person / _entering_person: 正在进/出电梯的人的列表
      _direction： 电梯的方向

    部分关键接口：
      run_elevator()
        运行逻辑：
          1. check当前所有target_floors和调度目标，决定一个合适楼层目标
          2. 管理当前电梯方向： 如果没有target_floors，电梯失去方向，可以任意调度
          3. 管理电梯的门：如果到达目标，要求开门
          4. 电梯动力学规划

      require_door_closing() / require_door_opening()
        请求电梯开/关门
        存在一些条件如下
          1. 电梯超载时，无法请求开门
          2. 有人进出时，无法关闭门
      
        run_elevator过程中，电梯会随时请求开门

      person_request_in（）：
        请求装载一个乘客
        如果请求不成功（现在有人还在下电梯/上电梯人太多/电梯太满等）返回失败

  mansion_config.py

    MansionConfig:大楼和电梯相关的配置，包含以下功能

      log_debug()/log_notice()/log_warning/log_fatal(): 不同级别日志打印，会带上日志的前缀
      step(): 演进一步时间
  
  mansion_manger.py:
    MansionManager: 整个大楼的管理者，模拟了大楼内出入人群以及电梯运作，一栋大楼可以有多部电梯，必须绑定一个MansionConfig
      run_mansion: 运行大楼，程序如下
        1. 利用PersonGenerator产生乘坐需求，出现乘坐需求响应的楼梯和方向（_wait_upward_persons_queue， _wait_downward_persons_queue）亮灯
        2. 运行电梯
        3. 电梯如果在一个位置停靠开门，就请求电梯装载乘客
        4. 如果有乘客等待时长超过5分钟，会放弃乘坐电梯
        5. 统计信息

  person_generators/:
    主要是模拟乘客产生的乘客产生器

dispatchers/:
  
  dispatcher_base.py
    电梯调度算法的基类，所有调度算法必须继承
    必须定义policy函数，输入为大楼的State,输出为ElevatorAction列表

  rule_benchmark_dispatcher.py
    规则调度，该规则调度逻辑如下
      1. 将所有电梯加入（闲置电梯 idle_ele_queue）队列
      2. 从idel_ele_queue队列中pop一个电梯
      3. 遍历电梯方向上可能达到的所有有需求楼层，如果该楼层未被处理而且其楼层的priority小于当前电梯的priority，就把这个楼层分配给当前电梯，更新priority，并且把原来分配的电梯加入idel_ele_queue
      4. 如果当前电梯方向一致的需求楼层不存在，将相反方向上，最远端的楼层作为调度目标
      5. 回到2，直到idle_ele_queue为空即停止

  rl_benchmark_dispatcher
    RL对于每一步电梯单独控制，每部电梯的状态量包括：
      1. 当前电梯的状态，包括楼层，速度，载重等等
      2. 其他电梯状态
      3. 电梯的序号
      4. 楼层上的乘坐需求情况

    每部电梯的Action包括
      DirectionIndicator * 调度的楼层层数 + 不进行任何动作 ( 2 * 调度的楼层层数+ 1 维度)

demo.py
  测试主流程
    1. 读配置
    2. 初始化Dispatcher
    3. 控制电梯，每一步都调用Dispatcher的policy函数，并且调用feedback返回反馈数据
    4. Reward目前定义为：等待电梯的人的时间损失+ 电梯能量损耗 + 因为等待太长时间都不能等到电梯而放弃的人的惩罚
