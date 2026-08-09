#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <algorithm>

#include "data.h"
#include "timer.h"

// constants
const size_t WORD_SIZE = 16;

// typedefs
typedef Data<WORD_SIZE> Word;
typedef Data<2 * WORD_SIZE> BusData;

int main(int argc, char **argv)
{
  Timer timer;
  BusData data;
  uint16_t *dest = new uint16_t[10];

  // put float example
  std::cout << "put float: \n";

  size_t lsb = 0;
  size_t size = 19;
  float msb_value = 32.0;
  float value = 31.4578;
  float out_value = 0;

  timer.set_name("example");
  timer.start();
  {
    data.put<float>(value, lsb, size, msb_value);
    out_value = data.get<float>(lsb, size, msb_value);

    dest[0] |= data.to_mil();
    dest[1] |= (data.to_mil() >> WORD_SIZE);
  }
  timer.stop();

  std::cout << "in value = " << value << "\n";
  std::cout << "out value = " << out_value << "\n";
  std::cout << data << "\n";
  std::cout << "msb_value = " << msb_value << "\n";
  std::cout << "mil = " << data.to_mil() << "\n";
  std::cout << "dest = " << dest[0] << " " << dest[1] << "\n";
  std::cout << timer << "\n \n";

  // put data from pointer
  std::cout << "put from pointer: \n";

  data.reset();
  data = {dest, lsb, size};
  out_value = data.get<float>(lsb, size, msb_value);

  std::cout << "in value = " << value << "\n";
  std::cout << "out value = " << out_value << "\n \n";

  // put char example
  std::cout << "put char: \n";

  data.reset();
  std::string str = "dio";
  std::string out_str = "";

  for (size_t i = 0; i < str.size(); i++)
  {
    data.put<char>(str[i], i * 8, 8, 0);
    out_str += data.get<char>(i * 8, 8, 0);
  }

  std::cout << "in string = " << str <<  "\n";
  std::cout << "out string = " << out_str <<  "\n";
  std::cout << data << "\n";
  std::cout << "mil = " << data.to_mil() << "\n";

  return 0;
}

