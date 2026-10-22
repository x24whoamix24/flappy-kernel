// SPDX-License-Identifier: Apache-2.0
// Draws a Frame (frame.cuh), one pixel per thread, in the style of the classic
// mobile game: pixel-art birds, banded green pipes, and a flat sky over a band
// of clouds, a pale skyline and bushes. All art is drawn here, procedurally.
//
// The scene is a stack of layers, front to back:
//   score   the big outlined number at the top
//   birds   the AI's bird, then the main bird
//   ground  striped grass, then sand
//   pipes
//   scenery bushes, skyline and clouds, scrolling slower with distance
//   sky
// Each layer returns the pixel's colour, or kTransparent where it does not
// cover the pixel.
#pragma once

#include "../core/state.cuh"
#include "font.cuh"
#include "frame.cuh"

namespace cotenant {

// ---------------------------------------------------------------- colours

// Packs a colour the way the host reads the frame (RGBA bytes, opaque).
__host__ __device__ constexpr uint32_t rgb(int r, int g, int b) {
  return 0xff000000u | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r);
}
constexpr uint32_t kTransparent = 0; // never a real colour: those are opaque

constexpr uint32_t kInk = rgb(32, 28, 30); // outlines of birds, pipes, score

constexpr uint32_t kSky = rgb(113, 197, 207);
constexpr uint32_t kCloud = rgb(235, 252, 220);
constexpr uint32_t kBuilding = rgb(225, 240, 209);
constexpr uint32_t kBuildingEdge = rgb(178, 220, 206);
constexpr uint32_t kWindow = rgb(196, 230, 214);
constexpr uint32_t kBush = rgb(131, 227, 140);
constexpr uint32_t kBushEdge = rgb(98, 196, 108);

constexpr uint32_t kPipeLight = rgb(128, 208, 16);
constexpr uint32_t kPipeDark = rgb(0, 168, 0);

constexpr uint32_t kGroundEdge = rgb(84, 56, 71);
constexpr uint32_t kGrassLight = rgb(156, 230, 89);
constexpr uint32_t kGrassDark = rgb(117, 191, 53);
constexpr uint32_t kGrassShadow = rgb(84, 128, 32);
constexpr uint32_t kSandShadow = rgb(204, 172, 83);
constexpr uint32_t kSand = rgb(225, 214, 148);

constexpr uint32_t kScoreText = rgb(255, 255, 255);

// A crashed bird keeps its shape but turns a washed-out grey.
__device__ __forceinline__ uint32_t fade_to_grey(uint32_t col) {
  uint32_t const r = col & 0xff, g = (col >> 8) & 0xff, b = (col >> 16) & 0xff;
  uint32_t const grey = ((r + g + b) / 3 + 160) / 2;
  return rgb(grey, grey, grey);
}

// ---------------------------------------------------------------- text

// White text with a dark outline `outline` pixels wide, its top-left corner
// at (x0, y0).
__device__ __forceinline__ uint32_t outlined_text_pixel(char const *text,
                                                        int len,
                                                        int x0,
                                                        int y0,
                                                        int scale,
                                                        int outline,
                                                        int x,
                                                        int y) {
  int const width = len * kGlyphAdvance * scale;
  if (x < x0 - outline || x >= x0 + width + outline || y < y0 - outline ||
      y >= y0 + kGlyphH * scale + outline) {
    return kTransparent;
  }
  if (text_hit(text, len, x0, y0, scale, x, y)) {
    return kScoreText;
  }
  for (int dy = -outline; dy <= outline; dy += outline) {
    for (int dx = -outline; dx <= outline; dx += outline) {
      if (text_hit(text, len, x0, y0, scale, x + dx, y + dy)) {
        return kInk;
      }
    }
  }
  return kTransparent;
}

constexpr int kScoreY = 30, kScoreScale = 5, kScoreOutline = 2;

__device__ __forceinline__ uint32_t score_pixel(Frame const &f, int x, int y) {
  int const width = f.score_len * kGlyphAdvance * kScoreScale;
  return outlined_text_pixel(f.score,
                             f.score_len,
                             (kScreenW - width) / 2,
                             kScoreY,
                             kScoreScale,
                             kScoreOutline,
                             x,
                             y);
}

// "AI" above the AI's bird while it flies, so players can tell the birds
// apart. In constant memory: a local array indexed at run time would live in
// local memory.
__constant__ char kAiLabel[2] = {'A', 'I'};
constexpr int kLabelScale = 2, kLabelOutline = 1, kLabelGap = 5;

__device__ __forceinline__ uint32_t ai_label_pixel(Frame const &f,
                                                   int x,
                                                   int y) {
  if (!f.show_ai || !f.ai_alive) {
    return kTransparent;
  }
  // Centred over the bird; the last glyph's blank column is not counted.
  constexpr int kWidth = 2 * kGlyphAdvance * kLabelScale - kLabelScale;
  int const top = int(f.ai_y) - kLabelGap - kGlyphH * kLabelScale;
  return outlined_text_pixel(kAiLabel,
                             2,
                             kBirdX + (kBirdW - kWidth) / 2,
                             top,
                             kLabelScale,
                             kLabelOutline,
                             x,
                             y);
}

// ---------------------------------------------------------------- birds

// The bird, 17 x 12 pixels drawn at 2x, so it exactly fills its 34 x 24
// hitbox. Facing right. k outline, y body, l belly, w white (eye and wing),
// s wing shading, p pupil, o beak, r beak shading.
constexpr int kSpriteW = 17, kSpriteH = 12, kSpriteScale = 2;
static_assert(kSpriteW * kSpriteScale == kBirdW &&
                  kSpriteH * kSpriteScale == kBirdH,
              "the sprite must fill the bird's hitbox");
__constant__ char kBirdSprite[kSpriteH][kSpriteW + 1] = {
    "......kkkkk......", //
    "....kkyyyyykk....", //
    "...kyyyyyykwwk...", //
    "..kyyyyyyykwwpk..", //
    ".kyyyyyyyykwwpk..", //
    ".kssskyyyyykwwk..", //
    "kswwwskyyyyykkkk.", //
    "ksswssklllkoooook", //
    ".kssskllllkrrrrk.", //
    "..kkklllllkkkkk..", //
    "...klllllllk.....", //
    "....kkkkkkk......", //
};

struct BirdColours {
  uint32_t body, belly;
};
// The main bird: the player's, or the AI's in the single-game view.
constexpr BirdColours kMainColours{rgb(250, 204, 46), rgb(247, 160, 47)};
constexpr BirdColours kAiColours{rgb(120, 190, 255), rgb(70, 130, 230)};
constexpr uint32_t kBirdWhite = rgb(252, 252, 246);
constexpr uint32_t kWingShade = rgb(226, 212, 170);
constexpr uint32_t kBeak = rgb(250, 110, 60);
constexpr uint32_t kBeakShade = rgb(214, 70, 40);

__device__ __forceinline__ uint32_t sprite_colour(char c, BirdColours colours) {
  switch (c) {
  case 'k':
  case 'p':
    return kInk;
  case 'y':
    return colours.body;
  case 'l':
    return colours.belly;
  case 'w':
    return kBirdWhite;
  case 's':
    return kWingShade;
  case 'o':
    return kBeak;
  case 'r':
    return kBeakShade;
  default:
    return kTransparent;
  }
}

// A bird with its top edge at `top`, nose rotated down by `tilt` radians.
__device__ uint32_t bird_pixel(
    float top, float tilt, BirdColours colours, bool alive, int x, int y) {
  constexpr float kReach = 24.f; // no corner is farther from the centre
  float const cx = kBirdX + kBirdW * 0.5f, cy = top + kBirdH * 0.5f;
  float const dx = x + 0.5f - cx, dy = y + 0.5f - cy;
  if (fabsf(dx) > kReach || fabsf(dy) > kReach) {
    return kTransparent;
  }
  // Undo the rotation, then look the pixel up in the sprite.
  float s, c;
  __sincosf(tilt, &s, &c);
  float const lx = dx * c + dy * s, ly = -dx * s + dy * c;
  int const sx = int(floorf(lx / kSpriteScale + kSpriteW * 0.5f));
  int const sy = int(floorf(ly / kSpriteScale + kSpriteH * 0.5f));
  if (sx < 0 || sx >= kSpriteW || sy < 0 || sy >= kSpriteH) {
    return kTransparent;
  }
  uint32_t const col = sprite_colour(kBirdSprite[sy][sx], colours);
  return alive || col == kTransparent ? col : fade_to_grey(col);
}

__device__ __forceinline__ uint32_t main_bird_pixel(Frame const &f,
                                                    int x,
                                                    int y) {
  return f.show_main
             ? bird_pixel(f.course.y, f.main_tilt, kMainColours, true, x, y)
             : kTransparent;
}

__device__ __forceinline__ uint32_t ai_bird_pixel(Frame const &f,
                                                  int x,
                                                  int y) {
  return f.show_ai ? bird_pixel(f.ai_y, f.ai_tilt, kAiColours, f.ai_alive, x, y)
                   : kTransparent;
}

// ---------------------------------------------------------------- ground

// Bands by depth below kGroundY: an edge, striped grass, its shadow, the
// sand's shadow, then sand.
constexpr int kEdgeEnd = 2, kGrassEnd = 13, kGrassShadowEnd = 15;
constexpr int kSandShadowEnd = 18;
constexpr int kStripeW = 12; // diagonal grass stripes, scrolling with the pipes
// Only called below kGroundY, where the ground covers everything behind it.
__device__ __forceinline__ uint32_t ground_pixel(Frame const &f, int x, int y) {
  int const depth = y - kGroundY;
  if (depth < kEdgeEnd) {
    return kGroundEdge;
  }
  if (depth < kGrassEnd) {
    int const stripe = ((x + int(f.tick * kPipeSpeed) + depth) / kStripeW) & 1;
    return stripe ? kGrassDark : kGrassLight;
  }
  if (depth < kGrassShadowEnd) {
    return kGrassShadow;
  }
  if (depth < kSandShadowEnd) {
    return kSandShadow;
  }
  return kSand;
}

// ---------------------------------------------------------------- pipes

// Each pipe ends at its mouth (the gap's edge) in a cap kPipeCapH rows tall,
// kPipeCapOverhang wider on each side. Pipes and caps are outlined in ink.
constexpr int kPipeCapH = 24, kPipeCapOverhang = 3, kPipeOutlineW = 2;

// Vertical light and dark bands across a pipe (u from 0 at the left edge to 1
// at the right), which read as a lit cylinder.
__device__ __forceinline__ uint32_t pipe_band(float u) {
  bool const light = u < 0.06f || (u >= 0.10f && u < 0.14f) ||
                     (u >= 0.50f && u < 0.58f) || (u >= 0.62f && u < 0.80f) ||
                     u >= 0.86f;
  return light ? kPipeLight : kPipeDark;
}

__device__ __forceinline__ uint32_t pipe_pixel(Env const &course,
                                               int x,
                                               int y) {
  for (int i = 0; i < kNumPipes; i++) {
    float const px = course.px[i], gy = course.gy[i];
    bool const top = y < gy;
    if (!top && y < gy + kPipeGap) {
      continue; // in the gap
    }
    // Rows from the mouth: 0 is the row next to the gap, in either pipe.
    float const row = top ? gy - 1 - y : y - (gy + kPipeGap);
    bool const cap = row < kPipeCapH;
    float const x0 = cap ? px - kPipeCapOverhang : px;
    float const x1 = cap ? px + kPipeW + kPipeCapOverhang : px + kPipeW;
    if (x < x0 || x >= x1) {
      continue;
    }
    bool const side_edge = x < x0 + kPipeOutlineW || x >= x1 - kPipeOutlineW;
    bool const cap_edge =
        cap && (row < kPipeOutlineW || row >= kPipeCapH - kPipeOutlineW);
    if (side_edge || cap_edge) {
      return kInk;
    }
    return pipe_band((x - x0) / (x1 - x0));
  }
  return kTransparent;
}

// ---------------------------------------------------------------- scenery

// Behind the pipes, above the ground: bushes in front of a pale skyline in
// front of a band of clouds. Farther layers scroll more slowly, and all are
// drawn in 2 x 2 blocks to match the pixel-art birds.
constexpr int kBlock = 2;
constexpr int kCloudTop = kGroundY - 104;  // highest bump of the clouds
constexpr int kSkylineTop = kGroundY - 86; // tallest building
constexpr int kBushTop = kGroundY - 30;    // highest bump of the bushes

// Height at `x` of a row of overlapping round bumps: one bump centred in each
// `period`, with a radius of `radius` plus up to two `step`s chosen per bump.
// Neighbouring bumps are included so that small ones never leave a gap.
__device__ __forceinline__ float
bumps(int x, int period, float radius, float step) {
  int const i = x / period;
  float height = 0.f;
  for (int n = i - 1; n <= i + 1; n++) {
    float const d = float(x - n * period) - period * 0.5f;
    float const r = radius + float(((n * 7) % 3 + 3) % 3) * step;
    height = fmaxf(height, sqrtf(fmaxf(r * r - d * d, 0.f)));
  }
  return height;
}

__device__ __forceinline__ uint32_t scenery_pixel(long long tick,
                                                  int x,
                                                  int y) {
  int const bx = x / kBlock * kBlock, by = y / kBlock * kBlock;
  int const scroll = int(tick);

  // Bushes: one pixel per tick.
  int const bush_x = bx + scroll;
  float const bush_top = kBushTop + 16 - bumps(bush_x, 18, 10.f, 3.f);
  if (by >= bush_top) {
    return by < bush_top + kBlock ? kBushEdge : kBush;
  }
  // Skyline: buildings 18 px wide with a 2 px gap, half speed.
  int const sky_x = bx + scroll / 2;
  int const building = sky_x / 20, lx = sky_x % 20;
  int const building_top = kSkylineTop + ((building * 53) % 5) * 10;
  if (lx < 18 && by >= building_top) {
    if (lx < kBlock || lx >= 18 - kBlock || by < building_top + kBlock) {
      return kBuildingEdge;
    }
    bool const window = lx >= 4 && lx < 14 && (lx - 4) % 4 < 2 &&
                        by >= building_top + 4 &&
                        (by - building_top - 4) % 6 < 2;
    return window ? kWindow : kBuilding;
  }
  // Clouds: a quarter speed.
  float const cloud_top =
      kCloudTop + 26 - bumps(bx + scroll / 4, 36, 20.f, 6.f);
  if (by >= cloud_top) {
    return kCloud;
  }
  return kSky;
}

// ---------------------------------------------------------------- frame

__device__ uint32_t shade(Frame const &f, int x, int y) {
  // The birds share one x. The player's bird is drawn over the AI's bird and
  // its label, so players always see their own.
  if (uint32_t const c = score_pixel(f, x, y)) {
    return c;
  }
  if (uint32_t const c = main_bird_pixel(f, x, y)) {
    return c;
  }
  if (uint32_t const c = ai_label_pixel(f, x, y)) {
    return c;
  }
  if (uint32_t const c = ai_bird_pixel(f, x, y)) {
    return c;
  }
  if (y >= kGroundY) {
    return ground_pixel(f, x, y);
  }
  if (uint32_t const c = pipe_pixel(f.course, x, y)) {
    return c;
  }
  return scenery_pixel(f.tick, x, y);
}

// Writes the frame into `fb` and makes it visible to the host. Called by
// every thread of the block.
__device__ void render_frame(Frame const &f, uint32_t *fb) {
  for (int pix = threadIdx.x; pix < kScreenW * kScreenH; pix += kThreads) {
    fb[pix] = shade(f, pix % kScreenW, pix / kScreenW);
  }
  __threadfence_system();
  __syncthreads();
}

} // namespace cotenant
