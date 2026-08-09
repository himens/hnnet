#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <cstring>
#include <byteswap.h>
#include <bitset>

#ifndef _DATA_H_
#define _DATA_H_

// MIL type definition
typedef uint64_t mil_t;

////////////////
// Data class //
////////////////
template <size_t N>
class Data 
{
  public:
    static const size_t SIZE = N;
    typedef std::bitset<SIZE> bitset_t;

    // constructors
    Data() {}

    template <class T>
      Data(const T &data) 
      {
        check_data_type<T>();

        _data = data; 
      }  
     
    template <class T>
      Data(const T *data, const size_t lsb, const size_t size) 
      {
        check_data_type<T>();
	check_bit_range(lsb, size);

        const size_t size_of_T = 8 * sizeof(T);
        const size_t number_of_elements = (size / size_of_T) + (size % size_of_T > 0);

        for (size_t i = 0; i < number_of_elements; i++)
        {
	  bitset_t bits = data[i]; 	
          _data |= (bits << i * size_of_T);
        }

        _data &= get_mask(lsb, size);
      }

    // print data
    friend std::ostream& operator<<(std::ostream& os, const Data &data) 
    {
      os << "Data: " << data.get_data();

      return os;
    }

    // get data 
    bitset_t get_data() const { return _data; }

    // reset data bits to zero
    void reset() { _data.reset(); }

    // swap data bytes
    void swap_bytes() 
    { 
      if      (SIZE == 16) _data = __bswap_16(to_mil()); 
      else if (SIZE == 32) _data = __bswap_32(to_mil()); 
      else if (SIZE == 64) _data = __bswap_64(to_mil()); 
      else 
      {
        std::cout << "[WARNING] Byte swap not supported for SIZE = " << SIZE << "! \n"; 
      }
    } 

    // reverse bits 
    void reverse_bits(const size_t lsb = 0, const size_t size = SIZE) 
    {
      check_bit_range(lsb, size);

      for (size_t i = 0; i < (size / 2); i++)
      {
	const size_t msb = lsb + size - 1;
	const bool bit = _data[lsb + i];

	_data[lsb + i] = _data[msb - i];
	_data[msb - i] = bit;
      }
    }

    // get data to MIL type 
    mil_t to_mil() const { return _data.to_ulong(); }

    // get data to type T
    template <class T>
      T get(const size_t lsb, const size_t size, const float msb_value) const 
      {
        check_data_type<T>(); 
	check_bit_range(lsb, size);

	bitset_t bits = _data & get_mask(lsb, size); // extract bits 
	bits >>= lsb; // shift back to bit 0

	const bool is_signed = msb_value < 0;
	const float lsb_value = get_lsb_value(size, msb_value);
	const mil_t mil = bits.to_ulong();
	T data = to_int(mil, size, is_signed) * lsb_value;

	return data;
      }

    // put data to word in MIL format
    template <class T>
      void put(const T &data, const size_t lsb, const size_t size, const float msb_value)
      {
        check_data_type<T>();
	check_bit_range(lsb, size);

	const bool is_signed = msb_value < 0;
	const float lsb_value = get_lsb_value(size, msb_value);
	const int scaled_data = data / lsb_value;
	const mil_t mil = to_mil(scaled_data, size, is_signed);

	bitset_t bits = mil;
	bits = (bits << lsb) & get_mask(lsb, size); // shift bits by lsb and mask the rest

	_data |= bits;
      }

  protected:
    // check data type 
    template <class T>
      void check_data_type() const 
      {
        static_assert(std::is_arithmetic<T>::value, "Input data type not arithmetic!");
        static_assert(sizeof(T) <= sizeof(mil_t), "Size of input data greater than mil_t!");
      }

    // check bit range 
    void check_bit_range(const size_t lsb, const size_t size) const
    {
      const size_t msb = lsb + size - 1;

      if (lsb > msb) throw std::range_error("Data::check_bit_range: lsb greater than msb!");
      if (lsb < 0 || msb > (SIZE - 1)) throw std::out_of_range("Data::check_bit_range: lsb or msb out-of-range!");
    }

    // get mask 
    bitset_t get_mask(const size_t lsb, const size_t size) const
    {
      check_bit_range(lsb, size);

      bitset_t mask = ~0; // all bits to 1
      mask = (mask >> (SIZE - lsb - size)) & (mask << lsb); // set bits above msb and below lsb to 0

      return mask;
    }

    // get lsb value
    float get_lsb_value(const size_t size, const float msb_value) const
    {
      if (size == 0) throw std::invalid_argument("Data::get_lsb_value: size is zero!");

      float lsb_value = msb_value != 0 ? std::abs(msb_value) / pow2(size - 1) : 1;

      return lsb_value;
    }

    // convert data from int to mil_t 
    mil_t to_mil(int value, const size_t size, const bool is_signed) const
    {
      if (size == 0) throw std::invalid_argument("Data::to_mil: size is zero!");

      const int min = is_signed ? -pow2(size - 1) : 0; 
      const int max = is_signed ? pow2(size - 1) - 1 : pow2(size) - 1; 

      if (value < min) value = min;
      if (value > max) value = max;

      mil_t mil = value < 0 ? value + pow2(size) : value;

      return mil;
    }

    // convert data from mil_t to int 
    int to_int(mil_t mil, const size_t size, const bool is_signed) const
    {
      if (size == 0) throw std::invalid_argument("Data::to_int: size is zero!");

      const mil_t positive_int_max = pow2(size - 1) - 1;
      const bool is_negative = is_signed && mil > positive_int_max;
      const mil_t min = is_negative ? positive_int_max + 1 : 0; 
      const mil_t max = !is_signed || is_negative ? pow2(size) - 1 : positive_int_max; 

      if (mil < min) mil = min;
      if (mil > max) mil = max;

      int value = is_negative ? mil - pow2(size) : mil;

      return value;
    }

    // fast power of 2 
    unsigned int pow2(const size_t size) const { return (1 << size); }

    // data members
    bitset_t _data;  
};
#endif
