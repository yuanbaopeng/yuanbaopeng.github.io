---
title: redis的bitfield命令的写属性
date: 2019-01-03 20:49:21
tags: redis
categories: redis
---

## BITFIELD命令介绍

redis支持一个按位操作的命令：`BITFIELD`

具体使用方法如下：

`BITFIELD key [GET type offset] [SET type offset value] [INCRBY type offset increment][OVERFLOW WRAP|SAT|FAIL]`

以下是 `BITFIELD` 命令支持的子命令：

- `GET <type> <offset>` —— 返回指定的二进制位范围。
- `SET <type> <offset> <value>` —— 对指定的二进制位范围进行设置，并返回它的旧值。
- `INCRBY <type> <offset> <increment>` —— 对指定的二进制位范围执行加法操作，并返回它的旧值。用户可以通过向 `increment` 参数传入负值来实现相应的减法操作。

除了以上三个子命令之外， 还有一个子命令， 它可以改变之后执行的 `INCRBY` 子命令在发生溢出情况时的行为：

- `OVERFLOW [WRAP|SAT|FAIL]`

当被设置的二进制位范围值为整数时， 用户可以在类型参数的前面添加 `i` 来表示有符号整数， 或者使用 `u` 来表示无符号整数。 比如说， 我们可以使用 `u8` 来表示 8 位长的无符号整数， 也可以使用 `i16` 来表示 16 位长的有符号整数。

## BITFIELD的写属性

在redis中，每一个命令都都被事先标记为是写命令还是读命令。

而BITFIELD命令却被标记为写，导致在有主从redis时，从redis被设置为readonly时，无法执行BITFIELD命令，无论子命令是GET还是SET。

在从redis使用该命令时必须将readonly属性设置为false。

具体在redis代码中

`{"bitfield",bitfieldCommand,-2,"wm",0,NULL,1,1,1,0,0}`

w：表示为write，写命令

m：表示为deny-oom，reject command if currently OOM



参考链接：

[https://redis.io/commands/command](https://redis.io/commands/command)

[http://redisdoc.com/string/bitfield.html](http://redisdoc.com/string/bitfield.html)

[http://redis-db.2338650.n4.nabble.com/BITFIELD-does-not-accept-reads-as-a-slave-td4941.html](http://redis-db.2338650.n4.nabble.com/BITFIELD-does-not-accept-reads-as-a-slave-td4941.html)

