// SPDX-License-Identifier: Apache-2.0
// The 5x7 digits the frame's score is drawn with, and decimal formatting of
// counts.
#pragma once

#include <cstdint>

namespace cotenant {

constexpr int kGlyphW = 5, kGlyphH = 7;
constexpr int kGlyphAdvance = kGlyphW + 1; // one blank column between glyphs

// The digits 0-9 in the classic 5x7 character-ROM style of HD44780-type LCD
// controllers. One byte per row, the leftmost pixel in bit 4.
__constant__ uint8_t kDigits[10][kGlyphH] = {
    {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}, // 0
    {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, // 1
    {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}, // 2
    {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}, // 3
    {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, // 4
    {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}, // 5
    {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}, // 6
    {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}, // 7
    {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}, // 8
    {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}, // 9
};

// True if pixel (x, y) is lit by the decimal `digits` drawn with their
// top-left corner at (x0, y0), each font pixel `scale` screen pixels wide.
__device__ __forceinline__ bool digits_hit(
    char const *digits, int len, int x0, int y0, int scale, int x, int y) {
  int const dx = x - x0, dy = y - y0;
  int const advance = kGlyphAdvance * scale;
  if (dx < 0 || dy < 0 || dy >= kGlyphH * scale || dx >= len * advance) {
    return false;
  }
  int const col = (dx % advance) / scale, row = dy / scale;
  if (col >= kGlyphW) {
    return false;
  }
  return (kDigits[digits[dx / advance] - '0'][row] >> (kGlyphW - 1 - col)) & 1;
}

// A count as shown on screen: NaN and negatives show 0, nine digits at most.
__device__ __forceinline__ long long display_count(double v) {
  return v >= 0.0 ? (long long)fmin(v, 999999999.0) : 0;
}

// Writes display_count(v) in decimal to dst, keeping its most significant
// digits if more than `room` would be needed. Returns the digits written.
__device__ int write_count(double v, char *dst, int room) {
  uint32_t x = uint32_t(display_count(v)); // fits: nine digits at most
  int digits = 1;
  for (uint32_t rest = x / 10; rest > 0; rest /= 10) {
    digits++;
  }
  int const written = min(digits, room);
  for (int i = digits - 1; i >= 0; i--) {
    if (i < written) {
      dst[i] = char('0' + x % 10);
    }
    x /= 10;
  }
  return written;
}

} // namespace cotenant
