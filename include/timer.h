#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <thread>    
#include <chrono>

#ifndef TIMER_H
#define TIMER_H

/////////////////
// Timer class //
/////////////////
class Timer
{
  public:
    Timer(const std::string name) { set_name(name); }
    Timer() {};

    static void sleep_ms(const unsigned int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
    static void sleep_s(const unsigned int s) { std::this_thread::sleep_for(std::chrono::seconds(s)); }

    friend std::ostream& operator<<(std::ostream& os, const Timer t) 
    {
      auto elapsed_time_us = t.get_elapsed_time_us();

      os << "[Timer " + t.get_name() + "] Elapsed time = ";
      if (elapsed_time_us < 1e3) os << elapsed_time_us << "us";
      else if (elapsed_time_us >= 1e3 && elapsed_time_us < 1e6) os << elapsed_time_us * 1e-3 << "ms";
      else os << elapsed_time_us * 1e-6 << "s";

      return os;
    }
    
    void start() { _start = std::chrono::system_clock::now(); }
    void stop() { _end = std::chrono::system_clock::now(); }
    void reset() { _start = _end = {}; }
    void set_name(const std::string name) { _name = name; }
    std::string get_name() const { return _name; }
    double get_elapsed_time_us() const { return get_elapsed_time_s() * 1e6; }
    double get_elapsed_time_ms() const { return get_elapsed_time_s() * 1e3; }
    double get_elapsed_time_s() const
    { 
      std::chrono::duration<double> elapsed_seconds = _end - _start;

      return elapsed_seconds.count();
    }

  private:  
    std::string _name;
    std::chrono::time_point<std::chrono::system_clock> _start;
    std::chrono::time_point<std::chrono::system_clock> _end;
};
#endif

