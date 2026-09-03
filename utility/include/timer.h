#pragma once
#include <thread>    
#include <chrono>

/////////////////
// Timer class //
/////////////////
// This class is used to measure the elapsed time. It provides methods to start, stop, and reset the timer,
// as well as to retrieve the elapsed time in microseconds, milliseconds, or seconds. 
// Additionally, it includes static sleep functions to pause execution for a specified duration.
// If timer started and stopped multiple times, the elapsed time will be the sum of all intervals between start and stop calls,
// until the timer is reset.
class Timer {
	public:
	    // Sleep functions
    	static void sleep_ms(const unsigned int ms) { 
			std::this_thread::sleep_for(std::chrono::milliseconds(ms)); 
		}
    	static void sleep_s(const unsigned int s) { 
			std::this_thread::sleep_for(std::chrono::seconds(s)); 
		}
        // Start, stop, reset timer
    	void start() {
			if (not _running) {
				_start = std::chrono::system_clock::now() - (_end - _start);
				_running = true;
			}
    	}
    	void stop() {
			if (_running) {
				_end = std::chrono::system_clock::now();
				_running = false;
			}
    	}
    	void reset() {
			if (not _running) {
      			_start = _end = {};
			}
		    else {
				_start = std::chrono::system_clock::now();
			}
    	}
		// Get elapsed time
    	double get_elapsed_time_us() const { 
			return get_elapsed_time_s() * 1e6; 
		}
    	double get_elapsed_time_ms() const { 
			return get_elapsed_time_s() * 1e3; 
		}
    	double get_elapsed_time_s() const { 
            const auto end = _running ? std::chrono::system_clock::now() : _end;
            const std::chrono::duration<double> elapsed_s{end - _start};
      		return elapsed_s.count();
    	}
  	private:  
	    // Data members
		bool _running{false};
    	std::chrono::time_point<std::chrono::system_clock> _start;
    	std::chrono::time_point<std::chrono::system_clock> _end;
};
