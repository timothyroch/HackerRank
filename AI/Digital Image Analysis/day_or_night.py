"""
Digital Camera Autodetect: Day or Night — Problem & Approach Notes
------------------------------------------------------------------

Problem (HackerRank):
- Task: Build a feature for a digital camera that automatically detects
  whether a photo is being taken in daytime or nighttime, and outputs
  "day" or "night".
- Input: A 2D grid of pixels, each pixel given in text as "B,G,R"
  (values 0–255), rows separated by newlines, pixels in a row separated by spaces.
- Output: A single word — either "day" or "night".
- Constraints: Input images <= 1 MB. The problem only requires handling
  *obviously* day vs. night scenes (not fuzzy cases like dawn/dusk/overcast).

My approach:
1. Brightness (luminance) baseline:
   - I compute per-pixel luminance (Y) using the BT.601 formula:
     Y = 0.114*B + 0.587*G + 0.299*R.
   - If the average luminance is very high, I immediately output "day".
   - If it’s very low, I output "night".

2. Mid-range cases:
   - Because some daytime photos are darker (shade, cloudy) and some nighttime photos are brighter (streetlights, neon),
     I add color cues to avoid misclassification.
   - I compute average Red and Blue; daytime scenes often have more blue/green (sky, foliage), while nighttime scenes tend 
     to have warm red/yellow lights.
   - I count “sky-like” pixels (bluish + greenish, with B > R).
   - I count “warm-light” pixels (reddish, bright, typical of night lamps).

3. Decision rules:
   - If luminance is extreme, I decide directly.
   - Otherwise, I use “votes” from blue dominance, sky-like fraction, bright-pixel fraction, and warm-light fraction.
   - If there’s a tie, I default to "day" when the scene leans blue; otherwise "night".

"""

from __future__ import print_function
import sys

# --------------------------------------------------------
# Helper: parse a pixel token "B,G,R" into three integers.
# --------------------------------------------------------
def parse_pixel(tok):
    try:
        b, g, r = tok.split(',')
        return int(b), int(g), int(r) 
    except:
        return None

# --------------------------------------------------------
# Main program logic
# Output: Prints one word ("day" or "night") to stdout
# --------------------------------------------------------
def main():
    # Running totals for averages
    total_Y = 0.0
    total_B = 0.0
    total_G = 0.0
    total_R = 0.0

    # Counters for heuristic features
    bright_count = 0       # how many pixels are bright overall
    sky_like_count = 0     # blue/green dominant pixels (typical daylight)
    warm_like_count = 0    # reddish pixels (typical warm lamps at night)
    n = 0                  # number of pixels processed

    # Threshold constants (tuned for clear day/night separation)
    MEAN_Y_DAY_HARD = 120.0     # very bright average luminance
    MEAN_Y_NIGHT_HARD = 50.0    # very dark average luminance
    PIXEL_BRIGHT_Y = 100.0      # per-pixel brightness cutoff
    BLUE_DOM_MARGIN = 12.0      # avgB must exceed avgR by this margin

    # "Sky-like" thresholds
    SKY_B_MIN = 110
    SKY_G_MIN = 80
    SKY_B_OVER_R = 15

    # "Warm-light" thresholds
    WARM_R_MIN = 140
    WARM_R_OVER_B = 20
    WARM_MIN_Y = 80

    # --------------------------------------------------------
    # Read pixel grid from stdin
    # --------------------------------------------------------
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        for tok in line.split():
            px = parse_pixel(tok)
            if px is None:
                continue
            b, g, r = px

            # Compute luminance (BT.601)
            Y = 0.114 * b + 0.587 * g + 0.299 * r

            # Accumulate totals
            total_Y += Y 
            total_B += b
            total_G += g
            total_R += r

            # Count bright pixels
            if Y >= PIXEL_BRIGHT_Y:
                bright_count += 1

            # Count sky-like pixels (blue+green dominant)
            if (b >= SKY_B_MIN) and (g >= SKY_G_MIN) and (b >= r + SKY_B_OVER_R):
                sky_like_count += 1

            # Count warm-light pixels (red dominant, bright)
            if (r >= WARM_R_MIN) and (r >= b + WARM_R_OVER_B) and (Y >= WARM_MIN_Y):
                warm_like_count += 1

            # Pixel counter
            n += 1

    # Edge case: no pixels read
    if n == 0:
        print("day")
        return

    # --------------------------------------------------------
    # Compute averages and fractions
    # --------------------------------------------------------
    mean_Y = total_Y / n
    avg_B  = total_B / n
    avg_R  = total_R / n
    frac_bright   = float(bright_count) / n
    frac_sky_like = float(sky_like_count) / n
    frac_warm     = float(warm_like_count) / n

    # --------------------------------------------------------
    # Hard decision by average luminance
    # --------------------------------------------------------
    if mean_Y >= MEAN_Y_DAY_HARD:
        print("day"); return
    elif mean_Y <= MEAN_Y_NIGHT_HARD:
        print("night"); return

    # --------------------------------------------------------
    # Mid-range cases: use heuristic "votes"
    # --------------------------------------------------------
    blue_dominant = (avg_B >= avg_R + BLUE_DOM_MARGIN)

    # Day votes: blue dominance with sky-like pixels, or many bright pixels
    day_vote = (blue_dominant and frac_sky_like >= 0.08) or (frac_bright >= 0.45)

    # Night votes: many warm lights, or very few bright pixels without blue dominance
    night_vote = (frac_warm >= 0.30) or (frac_bright <= 0.20 and not blue_dominant)

    # Final decision
    if day_vote and not night_vote:
        print("day")
    elif night_vote and not day_vote:
        print("night")
    else:
        # Tie-breaker: lean on blue dominance
        print("day" if blue_dominant else "night")

if __name__ == "__main__":
    main()
