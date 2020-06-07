---
title: C++的名字重整去除
date: 2019-01-02 21:25:20
tags: C++
categories: C++
---
C++为了支持同名函数，重载等功能，会将名字进行重整（mangle），在ABI（Application Binary Interface）中名字是重整过的，可以通过内部函数`abi::__cxa_demangle`实现还原，去除重整

具体实现可参考：

```c++
#include <exception>
#include <iostream>
#include <cxxabi.h>

struct empty { };

template <typename T, int N>
  struct bar { };


int main()
{
  int     status;
  char   *realname;

  // exception classes not in <stdexcept>, thrown by the implementation
  // instead of the user
  std::bad_exception  e;
  realname = abi::__cxa_demangle(e.what(), 0, 0, &status);
  std::cout << e.what() << "\t=> " << realname << "\t: " << status << '\n';
  free(realname);


  // typeid
  bar<empty,17>          u;
  const std::type_info  &ti = typeid(u);

  realname = abi::__cxa_demangle(ti.name(), 0, 0, &status);
  std::cout << ti.name() << "\t=> " << realname << "\t: " << status << '\n';
  free(realname);

  return 0;
}
  
```

代码会输出：

```shell
      St13bad_exception       => std::bad_exception   : 0
      3barI5emptyLi17EE       => bar<empty, 17>       : 0
```

备注：

去除重整的函数是使用C语言实现的，所以要记得调用free是释放内存。



参考链接：

[https://gcc.gnu.org/onlinedocs/libstdc++/manual/ext_demangling.html](https://gcc.gnu.org/onlinedocs/libstdc++/manual/ext_demangling.html)

[http://www.cnblogs.com/lizhenghn/p/3661643.html](http://www.cnblogs.com/lizhenghn/p/3661643.html)



