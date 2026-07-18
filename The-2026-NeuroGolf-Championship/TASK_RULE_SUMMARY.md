# NeuroGolf 2026: Per-task ARC rule summary (all 400)

Generated 2026-07-14 by 8 parallel agents, each independently reading  and verifying the rule against real train examples (not trusting the stale  DSL guesses). Intended as raw material for spotting simple-transform (identity/transpose/rotate/mirror/crop) full-score candidates.

### task001
- Rule: Treats the 3x3 input as a mask; tiles a 3x3 grid of 3x3 blocks where block (i,j) is a copy of the whole input pattern if input cell (i,j) is non-background, else a blank (all-zero) block.
- Category: tile-repeat
- Shape: constant 3x3 -> 9x9

### task002
- Rule: Any background (0) cell that sits directly between two cells of color 3 (immediate horizontal or vertical neighbors both 3) is recolored to 4; everything else is unchanged.
- Category: recolor-conditional
- Shape: variable (input=output same size per example, but that size differs across examples: 6x6, 10x10, 20x20)

### task003
- Rule: Recolors every 1 to 2, then extends the grid vertically from 6 rows to 9 rows by continuing the vertical repeating striped pattern (a short repeating sequence of rows) downward.
- Category: other-complex
- Shape: constant 6x3 -> 9x3

### task004
- Rule: Each grid contains one or two hollow diamond/parallelogram outline shapes (drawn in a single color) that are drawn slightly mis-aligned; each shape is redrawn shifted so it forms a properly symmetric diamond outline, correcting the distortion, while the background stays unchanged.
- Category: symmetry-fill
- Shape: variable (input=output same size per example; sizes seen: 8x9, 10x10, 14x9)

### task005
- Rule: Finds small colored motifs (e.g. a 3x3 block plus an adjacent single-color marker line) and extends each motif into a periodic repeating stripe pattern (using the marker's color) that fills the rest of its row or column.
- Category: object-detect
- Shape: constant 21x21 -> 21x21

### task006
- Rule: The grid is split by a single-color divider column into two equal-width halves; output cell is recolored to 2 where both halves have a non-background cell at the same relative position, otherwise 0 (logical AND overlay of the two halves).
- Category: symmetry-fill
- Shape: constant 3x7 -> 3x3

### task007
- Rule: A short sequence of colors is given along one corner/diagonal; the rule extends that diagonal color sequence periodically so every cell's color is determined by its diagonal index, filling the whole grid with repeating diagonal stripes.
- Category: tile-repeat
- Shape: constant 7x7 -> 7x7

### task008
- Rule: Detects two separate colored objects (a movable multi-cell shape and a fixed small square); translates the movable shape, keeping its form, until it becomes adjacent to the fixed square, which stays in place.
- Category: object-detect
- Shape: variable (input=output same size per example; sizes seen: 9x10, 11x10, 14x9)

### task009
- Rule: The grid is divided into repeating row-bands by full separator lines; small colored marks that appear in one band are carried down and extended into every subsequent band below it, accumulating/propagating along the columns as you move down the bands.
- Category: object-detect
- Shape: variable (input=output same size per example; sizes seen: 20x20, 23x23, 26x26)

### task010
- Rule: Cells of color 5 form nested square/L-shaped borders (concentric rings) around empty interiors; each ring is recolored by nesting depth from the outside in (outermost ring -> 1, next -> 2, then 3, then 4).
- Category: other-complex
- Shape: constant 9x9 -> 9x9

### task011
- Rule: The grid is split by divider lines into a 3x3 arrangement of 3x3 cells, each filled with a scattered set of several colors (one cell per color); a small subset of the 9 cells (selected by a non-obvious color/position matching rule) are flood-filled solid with one of their own colors, while every other cell is blanked to background.
- Category: other-complex
- Shape: constant 11x11 -> 11x11

### task012
- Rule: Each small plus/cross-shaped motif (a center cell of one color surrounded by 4 orthogonal arm cells of another color) is repeatedly stamped outward along the four cardinal directions from its own position, building a larger cross-of-crosses fractal pattern on the background.
- Category: other-complex
- Shape: constant 12x12 -> 12x12

### task013
- Rule: Two single marker cells (different colors) are placed somewhere in an otherwise blank grid; the rule builds a repeating two-color horizontal stripe pattern starting at the markers' columns and tiles that same row pattern identically down every row of the grid.
- Category: other-complex
- Shape: variable (each example a different size)

### task014
- Rule: The grid contains two stacked objects separated by a blank divider (an upper noisy pattern and a lower pattern); the rule crops out just the bounding box of the lower object's non-background pixels and discards everything else.
- Category: object-detect
- Shape: variable (each example a different size)

### task015
- Rule: Single seed pixels of specific colors spawn a small fixed stamp pattern around them (e.g. a color-2 seed produces a diagonal plus of 4s, a color-1 seed produces an orthogonal plus of 7s), with special overlap markings where two stamps would meet; the seeds themselves are kept.
- Category: other-complex
- Shape: constant 9x9 -> 9x9

### task016
- Rule: Recolors every cell using a fixed global color-to-color lookup table (e.g. 3->4, 2->6, 1->5, 8->9); the grid's geometric layout is unchanged, only the color of each cell is substituted.
- Category: recolor-map
- Shape: constant 3x3 -> 3x3

### task017
- Rule: The grid holds a large periodic "wallpaper" tiling pattern, but part of it has been corrupted/overwritten with wrong values or 0s; the rule restores the corrupted region using the correct repeating tile inferred from the rest of the pattern.
- Category: symmetry-fill
- Shape: constant 21x21 -> 21x21

### task018
- Rule: Several small noisy multi-color blobs are scattered on the grid; the rule keeps only specific blob(s) matching a selection criterion, crops to their bounding box, and discards the rest of the grid as background.
- Category: object-detect
- Shape: variable (each example a different size)

### task019
- Rule: The output is exactly double the input's height and width, filled with a checkerboard-like background pattern (introducing a new color not present in the input) into which repeated/transformed copies of the input's non-background marks are woven.
- Category: other-complex
- Shape: variable (output is always 2x the input's height and width, which itself varies per example)

### task020
- Rule: The grid contains a roughly symmetric diamond-shaped colored figure, but one part of it (e.g. the bottom arm) is missing a cell relative to its point-symmetric counterpart elsewhere in the figure; the rule fills in the missing cell(s) to restore the figure's symmetry.
- Category: symmetry-fill
- Shape: constant 10x10 -> 10x10

### task021
- Rule: The grid is divided by full-line dividers (one row and/or one column of a marker color) into rectangular cells; one of those cells is filled solid with a single uniform color, and the rule outputs just that solid rectangle at its own dimensions.
- Category: object-detect
- Shape: variable (each example a different size)

### task022
- Rule: Several small 1-2 cell colored objects are scattered around an 11x11 grid; the rule collects them and re-renders them compacted into a small 3x3 grid, preserving their relative row/column ordering.
- Category: object-detect
- Shape: constant 11x11 -> 3x3

### task023
- Rule: A single connected blob of color 5 has its outer-border cells recolored to 8 and its interior (non-border) cells recolored to 2, based purely on adjacency/border-vs-interior position within the connected shape.
- Category: recolor-conditional
- Shape: variable (input=output same size per example; sizes seen: 8x10, 9x11)

### task024
- Rule: A few isolated single-colored seed pixels each shoot out a full line across the whole grid: a color-2 seed produces a full vertical column of 2s through its column, while color-1 and color-3 seeds each produce a full horizontal row of their color through their row.
- Category: other-complex
- Shape: variable (input=output same size per example; sizes seen: 9x9, 10x8, 10x11, 12x11)

### task025
- Rule: The grid contains fixed vertical/horizontal divider lines and several small isolated marker pixels; each marker is pulled/duplicated so that it sits adjacent to (touching) the nearest divider line, effectively snapping loose markers onto the nearest wall.
- Category: object-detect
- Shape: variable (input=output same size per example; sizes seen: 15x14, 15x16, 18x19, 19x26)

### task026
- Rule: The grid contains noisy rows of colors 1/9; the rule locates the one row where a specific anomaly/gap pattern occurs and outputs a narrow 3-column strip marking that row's position with color 8, blanking all other rows.
- Category: other-complex
- Shape: constant 5x7 -> 5x3

### task027
- Rule: A wall-like maze pattern of color 1 has a small gap/opening; the rule floods a reachable interior/path region reachable through the opening and recolors it to 2, leaving the walls and unreachable background unchanged.
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task028
- Rule: One or two single seed pixels each spawn a ladder/grate-like pattern of horizontal bars with alternating gaps that grows outward from the seed to fill the rest of the grid in that seed's color.
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task029
- Rule: A large noisy grid is scanned to find and crop out a small distinctive sub-region (acting like a hidden key/legend), which is extracted and possibly recolored/simplified into the small output grid.
- Category: object-detect
- Shape: variable (each example a different size)

### task030
- Rule: Several small colored rectangular blocks float at different horizontal positions within one row-band; the rule slides them all horizontally until they are adjacent to each other, merging them into one contiguous row of blocks.
- Category: object-detect
- Shape: variable (input=output same size per example; sizes seen: 5x10, 10x10)

### task031
- Rule: Crops the grid to the bounding box of its non-background cells and returns that sub-grid unchanged (pure bounding-box crop, no recoloring).
- Category: fixed-crop
- Shape: variable (crop size depends on the shape's bounding box in each example)

### task032
- Rule: All non-background cells "fall" downward within their own column like gravity, stacking at the bottom of the grid in their original top-to-bottom order, leaving the vacated top cells as background.
- Category: other-complex
- Shape: variable (input=output same size per example; sizes seen: 4x4, 5x5, 6x6)

### task033
- Rule: The grid is a 3x3 arrangement of sub-grids separated by divider lines; one sub-grid contains a partial motif that is used as a key to complete/replicate a symmetric version of that motif into the matching positions of the other sub-grids sharing the same background frame.
- Category: symmetry-fill
- Shape: constant 17x17 -> 17x17

### task034
- Rule: A small seed block of two colors (e.g. mostly 4 with one 2) shoots out a diagonal 3-wide beam of the dominant color, extending from the seed to the grid's corner in the diagonal direction indicated by the seed's shape.
- Category: other-complex
- Shape: constant 9x9 -> 9x9

### task035
- Rule: A solid rectangular block sits among several isolated single-color marker pixels that are aligned with it along a row or column; each marker recolors the rectangle's border cell that faces it (nearest edge cell in that row/column) to the marker's own color, like an arrow pointing into the block.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task036
- Rule: A large mostly-random noisy grid contains one small non-random colored cluster hidden among the noise; the rule locates that cluster and outputs it (cropped/simplified) as a small grid.
- Category: object-detect
- Shape: variable (each example a different, much smaller, output size)

### task037
- Rule: Several isolated single-color seed pixels are scattered on the grid; each seed shoots out a diagonal ray of its own color extending outward toward the nearest grid corner/edge.
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task038
- Rule: Counts occurrences of a specific pattern (adjacent pair of color-1 cells) in the noisy grid and outputs that count as a run of 1s of that length in a fixed 1x5 output row, padded with 0s.
- Category: counting
- Shape: constant 9x9 -> 1x5

### task039
- Rule: The grid contains one object that is symmetric both horizontally and vertically; the rule crops to that object's bounding box and then keeps only its top-left quadrant (one quarter of the symmetric shape).
- Category: object-detect
- Shape: constant 10x10 -> 3x3

### task040
- Rule: The grid has two parallel full border lines (e.g. left/right columns or top/bottom rows) of two different colors, with several loose single-color marker pixels scattered between them; each marker is recolored to match whichever border line is nearest to it.
- Category: recolor-conditional
- Shape: constant 10x10 -> 10x10

### task041
- Rule: Several same-colored points are scattered along a diagonal/staircase arrangement; the rule flood-fills the triangular/staircase region between them so the diagonal line of points becomes a solid filled block of that color.
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task042
- Rule: Small pairs of diagonally-adjacent marker cells indicate a directional sequence; the rule extrapolates and stamps one additional point (in color 8) continuing that diagonal sequence one step further in empty space, for each marker pair found.
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task043
- Rule: A header row marks certain columns with color 5; any other row that already contains a lone marker (5) gets those same header columns additionally recolored to 2 (projecting the header's column positions down into marker rows), while the original markers and non-marker rows are left unchanged.
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task044
- Rule: The grid contains rectangular box-shaped objects (outlined/filled with color 5) that each have a small anomalous patch of extra color breaking their otherwise symmetric interior; the rule cleans the anomaly out of one box while relocating/restoring the corresponding symmetric pattern in another matching box.
- Category: symmetry-fill
- Shape: constant 10x10 -> 10x10

### task045
- Rule: Each row has a marker color in its leftmost and rightmost cell; if the two end markers are the same color, the entire row is filled solid with that color, otherwise the row is left unchanged.
- Category: recolor-conditional
- Shape: constant 10x10 -> 10x10

### task046
- Rule: A legend row encodes color/length information (marker cells of color 5 next to digits indicating run-lengths); the rule decodes this compressed legend into an expanded run-length bar of colored blocks in a narrower output grid.
- Category: other-complex
- Shape: variable (each example a different size; width shrinks from input to output)

### task047
- Rule: Two isolated single-color marker pixels each project a full cross (their entire row and column filled with their own color); at the two points where one marker's row crosses the other marker's column, that intersection cell is recolored to a special mixed color (2) instead.
- Category: other-complex
- Shape: constant 9x9 -> 9x9

### task048
- Rule: A noisy grid of scattered small colored blocks is analyzed to determine a single representative color (e.g. the color forming a distinguishing pattern/parity among the blocks), and the output is just that one color as a single 1x1 cell.
- Category: counting
- Shape: variable (input size varies per example; output is always 1x1)

### task049
- Rule: The grid contains several separate rectangular colored regions, some of which are noisy/broken internally and one of which is a fully solid uniform block; the rule detects and crops out that one solid rectangular region.
- Category: object-detect
- Shape: variable (each example a different size)

### task050
- Rule: Pairs of isolated same-color marker pixels that share a row or column are connected by drawing a straight line of a third connector color (3) between them, leaving the markers themselves and unrelated background untouched.
- Category: object-detect
- Shape: variable (input=output same size per example; sizes seen: 3x3 up to 12x13)

### task051
- Rule: Detects a small blob shape whose "tip" cell has a distinct color from the rest of the blob, then shoots a ray of that tip color outward (away from the blob) all the way to the grid edge.
- Category: object-detect
- Shape: variable (per-example square-ish grids, io shapes match but vary 10x15 to 16x11)

### task052
- Rule: For each row of the 3x3 grid, if the row is uniform (all three cells the same color), replace it with all 5s; otherwise replace it with all 0s.
- Category: recolor-conditional
- Shape: constant 3x3 -> 3x3

### task053
- Rule: Shifts the entire grid down by one row; the new top row becomes background (0) and the original bottom row is discarded.
- Category: other-complex
- Shape: constant 3x3 -> 3x3

### task054
- Rule: Each rectangle contains a small plus/cross-shaped seed near its center; rays are drawn outward from the seed's arms in the seed's colors, filling straight lines within the rectangle until they hit the rectangle border.
- Category: other-complex
- Shape: constant 30x30 -> 30x30

### task055
- Rule: The grid is divided into cells by horizontal/vertical separator lines (color 8); each resulting rectangular region is flood-filled with a color that depends on which row/column band it falls in.
- Category: other-complex
- Shape: variable (per-example grids differ: 12x14, 17x15, 18x19)

### task056
- Rule: Classifies the 3x3 input pattern (a specific arrangement of a color and background) into one of several categories and outputs a single pixel whose color encodes which pattern class it matched.
- Category: other-complex
- Shape: constant 3x3 -> 1x1

### task057
- Rule: Crops to the bounding box of the non-background shape, then tiles that cropped block twice side-by-side horizontally.
- Category: tile-repeat
- Shape: constant 8x8 -> 3x6

### task058
- Rule: Input is always a blank grid of size NxN; output is a fixed recursive/fractal decorative pattern of color 3 whose structure depends only on N.
- Category: other-complex
- Shape: variable (square N x N -> N x N, N varies 6,8,10,13,15,18)

### task059
- Rule: The grid is split by rows/columns of 5s into 9 cells, each containing scattered single-color marker dots; the cell with the most markers is filled solid with that marker's color and all other cells are cleared to background.
- Category: counting
- Shape: constant 11x11 -> 11x11

### task060
- Rule: A single row contains two colored markers; the row between them is filled with the left marker's color up to the midpoint and the right marker's color after it, with a special color 5 placed exactly at the midpoint.
- Category: other-complex
- Shape: constant 5x11 -> 5x11

### task061
- Rule: The grid contains a repeating periodic tile pattern that has been corrupted (zeroed out) in some rows/blocks; the output reconstructs the full periodic pattern by propagating it from the intact regions.
- Category: symmetry-fill
- Shape: constant 18x18 -> 18x18

### task062
- Rule: A plus/cross-shaped object has one arm tipped with a different (minority) color; that arm is extended further and the whole object is recolored to the majority color, while the background is recolored from 0 to 3.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task063
- Rule: A shape outlined by border colors encloses a background-filled interior region; the enclosed interior is flood-filled with color 3 while the border and exterior are left unchanged.
- Category: object-detect
- Shape: variable (square N x N -> N x N, N = 10, 12, or 14)

### task064
- Rule: A rectangle outline and an isolated single-pixel marker of a special color exist elsewhere in the grid; a straight line is drawn in the marker's color connecting the rectangle to the marker.
- Category: object-detect
- Shape: variable (grids range from 9x12 to 19x21)

### task065
- Rule: The grid is divided into 4 quadrants by cross-shaped separator lines; the output is whichever quadrant contains a color cell that differs from the other three (matching) quadrants.
- Category: object-detect
- Shape: variable (input NxN varies 5,7,11,13; output is roughly half that size)

### task066
- Rule: Two marker regions are connected by tracing a path (like a maze corridor) through the grid's open cells, drawn in a new color.
- Category: object-detect
- Shape: variable (square N x N -> N x N, N = 10, 13, 15, 20)

### task067
- Rule: The grid is a horizontal repetition of the same pattern block 3 times; the output crops out just one period (the first block).
- Category: tile-repeat
- Shape: variable (width = 3x height in all cases, e.g. 3x9->3x3, 4x12->4x4)

### task068
- Rule: Finds the one color that occurs exactly once in the grid; draws a 3x3 box (border color 2, center = that unique color) centered on its location and clears everything else to background.
- Category: counting
- Shape: constant 10x10 -> 10x10

### task069
- Rule: A small 2x2 "key" pattern of colors appears once in the grid, and several other locations are marked with 2x2 blocks of a placeholder color (8); each placeholder block is replaced with a copy of the key pattern, and the original key is erased.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task070
- Rule: A periodic striped/checker background pattern contains a single cell that breaks the periodicity (an anomalous color); a small cross/plus shape is highlighted in color 3 around each such anomaly.
- Category: symmetry-fill
- Shape: constant 17x17 -> 17x17

### task071
- Rule: A shape drawn with an outline color overlaps a solid-filled blob of another color; the overlapping filled region is stripped away/collapsed, compacting the outline shape.
- Category: object-detect
- Shape: constant 16x16 -> 16x16

### task072
- Rule: The grid is split into a top half and bottom half by a separator row of a fixed color; the two halves are overlaid and any cell that is non-background in either half becomes color 3 in the output (logical OR).
- Category: symmetry-fill
- Shape: constant 13x5 -> 6x5

### task073
- Rule: A single colored pixel "falls" straight down under gravity until it reaches a solid border row, where it merges into that row, replacing the border's color at that column.
- Category: other-complex
- Shape: constant 5x5 -> 5x5

### task074
- Rule: The grid is meant to be vertically mirror-symmetric but some rows near the middle are corrupted/mismatched; the output repairs those rows so the whole grid becomes a perfect vertical mirror of itself.
- Category: symmetry-fill
- Shape: constant 30x30 -> 30x30

### task075
- Rule: A small 3x3 "key" pattern sits at the top of a vertical spine; marker dots elsewhere along the spine indicate row-bands and column offsets where a copy of the key pattern should be stamped.
- Category: object-detect
- Shape: variable (14x15 or 13x13 depending on example)

### task076
- Rule: Several small shapes/markers exist in the grid; lines/connections are drawn in a new color extending from certain shapes toward nearby markers, effectively linking related objects.
- Category: object-detect
- Shape: variable (13x13 or 14x15 depending on example)

### task077
- Rule: A large blob of one color contains isolated noise dots of another color touching or near it; the connected component of the blob that touches a noise dot gets recolored to color 4.
- Category: object-detect
- Shape: variable (grids range from 15x14 to 17x18)

### task078
- Rule: A vertical trail of marker dots moves downward and merges into a maze-like obstacle shape, filling the matching gap in the shape and disappearing elsewhere (a falling/pouring simulation).
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task079
- Rule: A specific small motif (e.g. a checkerboard-like 3x3 pattern) appears twice among scattered noise dots in the grid; the output crops out just that repeated motif.
- Category: object-detect
- Shape: constant 14x14 -> 3x3

### task080
- Rule: The grid is a coarse grid of blocks separated by lines; blocks containing a special nested sub-pattern cause the same nested pattern to be recursively stamped into other related blocks (fractal-style propagation).
- Category: other-complex
- Shape: variable (square N x N -> N x N, N = 24, 27, 29, ...)

### task081
- Rule: Each L-tromino shape (3 cells of a color forming an L within a 2x2 box) has its missing 4th corner cell filled in with color 1, completing it into a full 2x2 square.
- Category: object-detect
- Shape: constant 7x7 -> 7x7

### task082
- Rule: A single row contains a few sparse colored dots; the output tiles the row pattern vertically down the grid, alternating between the original row and a shifted copy to create a brick-like periodic pattern.
- Category: tile-repeat
- Shape: variable (height 6, width varies 7/10/12 across examples)

### task083
- Rule: The small input pattern is mirrored horizontally and concatenated with itself, then that combined block is stacked twice vertically, doubling both dimensions.
- Category: tile-repeat
- Shape: constant 3x4 -> 6x8

### task084
- Rule: A border column keeps its original color; a diagonal line of color 2 is drawn from the top-right corner to the bottom-left, and the entire bottom row (except the border column) is filled with color 4.
- Category: other-complex
- Shape: variable (square N x N -> N x N, N = 3, 7, 10, 15)

### task085
- Rule: Solid horizontal rectangular bands of a color have their middle row's cells punched out into an alternating on/off (striped) pattern, leaving the rest of each band solid.
- Category: other-complex
- Shape: variable (grids 8x20, 8x30, or 11x20)

### task086
- Rule: A small nested-ring shape (outer ring of one color, center dot of another) grows outward by one layer, producing a larger set of concentric rings alternating between the two colors.
- Category: other-complex
- Shape: variable (square N x N -> N x N, N = 10 or 12)

### task087
- Rule: Rotates the entire grid 180 degrees.
- Category: rotate180
- Shape: constant 3x3 -> 3x3

### task088
- Rule: Crops to the bounding box of one colored shape and recolors all of its cells to match a different marker color found elsewhere in the input.
- Category: recolor-map
- Shape: variable (input/output pairs vary widely, e.g. 7x7->3x3, 12x18->4x8)

### task089
- Rule: An existing small stair-step shape is copied and translated so a new instance of the same shape ends at an isolated marker dot elsewhere in the grid.
- Category: object-detect
- Shape: constant 13x13 -> 13x13

### task090
- Rule: Two of the three rows share a common gap of background cells at the same column offset (row 0 is a boundary/key row that is left unchanged); that shared gap in the other two rows is recolored to color 6.
- Category: other-complex
- Shape: variable (3 or 4 rows tall, width 20-30)

### task091
- Rule: The grid contains a symmetric core pattern surrounded by extra isolated noise dots; the output crops out just the bounding box of the clean symmetric core, discarding the noise.
- Category: object-detect
- Shape: variable (input/output pairs vary, e.g. 9x9->5x5, 14x13->10x4)

### task092
- Rule: Pairs of isolated dots sharing the same color are connected by a straight line/path in that color, with a distinct color marking any crossing/junction point.
- Category: object-detect
- Shape: variable (grids are 20x10, 20x20, or 30x20)

### task093
- Rule: Isolated marker dots above and below a solid horizontal band are each connected to the band with a short line in the band's color, effectively projecting each dot onto the band.
- Category: object-detect
- Shape: constant 14x14 -> 14x14

### task094
- Rule: A small hollow square frame sits inside a background field; a cross-shaped ray is emitted from the frame's center row and column outward to the grid's edges in a new color.
- Category: object-detect
- Shape: constant 15x15 -> 15x15

### task095
- Rule: Around every isolated single-color dot in the grid, a 3x3 ring of color 1 is drawn, keeping the original dot's color at the center.
- Category: object-detect
- Shape: constant 9x9 -> 9x9

### task096
- Rule: Amid a large noisy patterned grid, there is a small symmetric core motif bordered by a distinctive color; the output crops out just that symmetric motif.
- Category: object-detect
- Shape: variable (input varies 13x17 to 19x19; output is 7x7 or 11x11)

### task097
- Rule: Removes all isolated single-pixel noise dots from the grid while keeping any multi-cell connected shapes unchanged.
- Category: object-detect
- Shape: variable (grids range from 11x19 to 17x14)

### task098
- Rule: Solid filled rectangles in the grid are hollowed out, keeping only their outline/border and clearing the interior to background.
- Category: object-detect
- Shape: variable (grids range from 8x7 to 17x19)

### task099
- Rule: A shape's outermost layer of cells is recolored to color 2 (like eroding/peeling one layer), while the interior structure of the shape is preserved.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task100
- Rule: Two hollow ring-shaped objects of different colors and sizes exist in the grid; the output is a solid 2x2 block filled with the color of whichever ring is larger.
- Category: counting
- Shape: constant 10x10 -> 2x2

### task101
- Rule: Finds pairs of lone marker-color dots that are vertically aligned in the same column and separated by a fixed gap matching a nearby template shape, then stamps a copy of the template's inner line pattern between each such pair of markers.
- Category: object-detect
- Shape: variable (in=out per example, e.g. 14x12->14x12 or 17x21->17x21, but the fixed size differs per example)

### task102
- Rule: Finds hollow rectangular outlines made of one color; if the fully-enclosed interior forms a solid rectangular block of background cells, recolors that interior block to a second color (irregular/non-rectangular enclosed holes are left unchanged).
- Category: object-detect
- Shape: constant 12x12 -> 12x12

### task103
- Rule: A small shape made of one color inside a 3x3 grid is classified by its geometric arrangement (e.g. diagonal vs. other pattern), and the whole output is a single 1x1 cell colored according to which shape-class was detected.
- Category: other-complex
- Shape: constant 3x3 -> 1x1

### task104
- Rule: A 3x3 input contains one "seed" cell plus a small cluster of a second color; the output stamps two solid squares (larger than the original cells) diagonally, anchored at the seed cell's coordinates and offset by one square-size further along the diagonal.
- Category: other-complex
- Shape: constant 3x3 -> 9x9

### task105
- Rule: Finds pairs of same-color marker dots aligned in the same row or column with only background cells between them, and fills the gap cells directly between each pair with a second color, drawing connecting line segments between the markers.
- Category: other-complex
- Shape: variable (in=out per example, heights range roughly 9-14, width fixed at 13)

### task106
- Rule: Tiles the input into a 2x2 mosaic where each tile is a mirrored variant of the original (original, horizontally mirrored, vertically mirrored, and both-mirrored), forming a symmetric kaleidoscope-style enlargement.
- Category: tile-repeat
- Shape: variable (2x2->4x4 or 3x3->6x6, always a 2x scale mirrored mosaic)

### task107
- Rule: Scales the input up by a variable factor and redraws its inner rectangular pattern proportionally, additionally marking the four corners of the scaled inner shape's bounding box with small diagonal corner-bracket markers of a new color.
- Category: other-complex
- Shape: variable (constant 5x5 input, output size varies 10x10 up to 30x30 depending on example)

### task108
- Rule: The input encodes a sparse 5x5 grid of colored points spaced 2 cells apart; the output re-renders that sparse point-grid at 4x zoom, drawing each point (or empty background) as a solid 4x4 block of its color.
- Category: other-complex
- Shape: constant 10x10 -> 20x20

### task109
- Rule: Removes a plus-shaped dividing line of one color that splits the grid into four quadrants, and fills all four (now merged) quadrants with mirror-symmetric copies of whichever pattern was in one quadrant, recolored to the divider's color; output is cropped to exclude the removed divider row/column.
- Category: symmetry-fill
- Shape: variable (NxN -> (N-1)x(N-1), N ranges roughly 7-13)

### task110
- Rule: Every background (color 0) cell is repainted with a color inferred from its surrounding pattern, effectively removing all background cells and inpainting them so only the non-background colors remain.
- Category: other-complex
- Shape: constant 29x29 -> 29x29

### task111
- Rule: A single marker-color dot points at one of several small disconnected shapes in the grid; the shape adjacent to the marker is cropped out and returned as the 3x3 output.
- Category: object-detect
- Shape: constant 10x10 -> 3x3

### task112
- Rule: A small 2x2 block of one color acts as a center marker; a nearby shape of another color is mirrored into the remaining three quadrants around that center, producing 4-fold symmetry, while the center marker itself is left unchanged.
- Category: symmetry-fill
- Shape: variable (in=out per example; sizes include 10x10, 12x14, 18x14, 20x30)

### task113
- Rule: Overlays the grid with its own vertical mirror image (flipped upside-down) and keeps any colored cell from either version, effectively duplicating a pattern near one edge symmetrically to the opposite edge.
- Category: symmetry-fill
- Shape: variable (in=out per example; fixed height 10, widths 3/5/6)

### task114
- Rule: Pads the grid by 1 cell on every side using edge-replication (nearest-neighbor padding), then sets the four outermost corner cells of the padded result to background, producing a rounded-corner enlarged copy.
- Category: fixed-pad
- Shape: variable (out = in+2 in each dimension, e.g. 2x2->4x4, 2x3->4x5, 3x3->5x5)

### task115
- Rule: Counts the distinct non-background colors present in the grid and outputs a thin 1-cell-wide vector (one cell per distinct color) summarizing which colors occurred, oriented to match the input's aspect ratio.
- Category: counting
- Shape: variable (e.g. 14x16->1x3, 9x7->3x1, 11x9->4x1)

### task116
- Rule: Creates a vertically mirrored copy of the input (flipped upside-down) and stacks it above the original input, doubling the grid's height.
- Category: symmetry-fill
- Shape: constant 3x4 -> 6x4

### task117
- Rule: A small cross/X shaped marker of one color indicates a center of symmetry; a nearby shape of a second color is mirrored into the other three quadrants around that center, producing 4-fold symmetric copies while the marker itself stays fixed.
- Category: symmetry-fill
- Shape: variable (in=out per example; sizes include 12x12, 14x14, 15x15)

### task118
- Rule: Finds plus/cross-shaped groups of one color whose arms have a single missing (gap) cell, and fills that gap cell with a new color, marking the inferred break point in an otherwise solid line.
- Category: object-detect
- Shape: variable (in=out per example; sizes include 11x12, 18x19, 19x22, 20x20, 20x22)

### task119
- Rule: Two adjacent marker cells define the start and direction of a diagonal ray; the ray is extended cell-by-cell in that direction, bouncing (reflecting) off a solid wall block, until it exits the grid.
- Category: other-complex
- Shape: constant 12x12 -> 12x12

### task120
- Rule: For each solid filled rectangle of a single color, keeps a 1-cell border ring of the original color and recolors the entire interior fill to a fixed new color, hollowing the rectangle out.
- Category: object-detect
- Shape: variable (in=out per example; sizes include 12x11, 12x13, 13x15, 14x13)

### task121
- Rule: Several small plus-shaped objects of different colors are scattered in the grid; one of them has a single mismatched center pixel (an impurity). That shape is selected, its center pixel corrected to match the surrounding color, and the corrected 3x3 patch is output.
- Category: object-detect
- Shape: constant 13x13 -> 3x3

### task122
- Rule: A small 3x3 frame shape with a dot at its center slides along a horizontal dotted guide-line so that its center lands on the next dot position along the line.
- Category: other-complex
- Shape: variable (in=out per example; sizes include 7x13, 7x17, 7x7, 13x7)

### task123
- Rule: Recursively extends/tiles the input's non-background pattern outward into a doubled-size output, growing a fractal-like staircase of repeated shapes so that no background cells remain.
- Category: other-complex
- Shape: constant 5x5 -> 10x10

### task124
- Rule: Repeats/extends the input pattern downward (tiling vertically, truncating as needed) until the output reaches a fixed height of 10 rows, keeping the width unchanged.
- Category: tile-repeat
- Shape: variable input height (5-8) x constant width 10 -> constant 10x10

### task125
- Rule: Draws a 1-cell border of a new color around every rectangular shape made of a given color, and additionally fills any fully-enclosed background hole inside one of those shapes with a second new color.
- Category: object-detect
- Shape: constant 15x15 -> 15x15

### task126
- Rule: A small U-shaped "cup" made of one color marks a column; a single dot of a second color is placed at the bottom row directly below the cup's opening, as if a marble fell straight down from it.
- Category: other-complex
- Shape: variable (in=out per example; sizes include 5x5, 5x7, 7x11, 8x8)

### task127
- Rule: Vertical divider lines of one color split the grid into panels, each containing a small numeric marker color; every panel is entirely flood-filled with a color equal to a fixed offset applied to its marker's color value.
- Category: recolor-conditional
- Shape: variable (in=out per example; fixed shapes 3x11 or 7x11)

### task128
- Rule: All colored shapes in the grid slide/fall upward (gravity toward the top edge) as far as possible until blocked by the grid boundary or another shape, keeping their relative column positions.
- Category: other-complex
- Shape: variable (in=out per example; sizes include 12x12, 14x14, 15x15)

### task129
- Rule: Finds the most common non-background color in the input and fills the entire output with that single color.
- Category: counting
- Shape: constant 3x3 -> 3x3

### task130
- Rule: Amid scattered single-cell noise, the grid contains a few distinct solid-colored blocks; the output is a small grid summarizing which colored blocks were found and their relative positions.
- Category: object-detect
- Shape: constant 9x9 -> 3x3

### task131
- Rule: A colored squiggle shape slides horizontally until it touches a fixed vertical line of a second color, and a new vertical marker line is drawn one column to the left of the shape's final resting position.
- Category: other-complex
- Shape: variable (in=out per example; sizes include 4x16, 4x18, 17x5)

### task132
- Rule: Two isolated same-colored dot markers are treated as opposite corners of a rectangle, and the entire rectangle between them is filled solid with that color.
- Category: object-detect
- Shape: variable (in=out per example; sizes include 6x11, 7x8, 9x8, 10x10)

### task133
- Rule: A small plus-shaped template (one color with a single arm replaced by a marker color) encodes which directions to stamp copies of a second block found elsewhere; copies of that block's main color are added in the matching arm positions, while the arm matching the marker color is left as-is.
- Category: other-complex
- Shape: variable (in=out per example; sizes include 16x12, 15x18, 16x18, 17x18, 19x30)

### task134
- Rule: Divides the grid (ignoring two large rectangular reference blocks) into a 3x3 layout of regions and outputs a 3x3 grid marking which regions contain a dense cluster of scattered noise-colored dots.
- Category: counting
- Shape: variable input (roughly 20x24 to 24x26) -> constant 3x3

### task135
- Rule: The grid contains many scattered single-colored dots of mostly-unique colors; the output is a small grid highlighting the colors that repeat, placed according to their relative positions.
- Category: counting
- Shape: constant 9x9 -> 3x3

### task136
- Rule: Each solid 2x2 colored block sprouts a thin one-cell-wide diagonal trail extending away from one of its corners toward the nearest grid edge, in the same color as the block.
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task137
- Rule: Scattered isolated dots each determine the size of a hollow square outline; the output draws a set of nested, concentric hollow square frames anchored at the corner, sized according to the positions of the original dots.
- Category: other-complex
- Shape: variable (constant per example; sizes 23x23 or 28x28)

### task138
- Rule: Divider lines of solid colors split the grid into a mosaic of cells containing repeated rows/columns; the output compresses each region by collapsing runs of identical rows/columns, keeping the divider lines, producing a smaller downsampled mosaic.
- Category: other-complex
- Shape: variable (e.g. 22x22->15x12, 12x12->8x8, 16x15->12x10)

### task139
- Rule: For each connected shape of a given color, finds its bounding box and recolors any background cells within that box (that aren't part of the shape itself) to a new color, filling in the shape's "holes."
- Category: object-detect
- Shape: constant 9x9 -> 9x9

### task140
- Rule: Rotates the grid 180 degrees.
- Category: rotate180
- Shape: constant 3x3 -> 3x3

### task141
- Rule: Draws both full diagonals (an X shape) passing through a single marked point, extending outward to the grid edges, in the same color as the marker.
- Category: other-complex
- Shape: variable (constant per example; sizes 7x7, 15x15, 17x17)

### task142
- Rule: Tiles the input into a 2x2 mosaic where each tile is a mirrored variant of the original (original, horizontally mirrored, vertically mirrored, and both-mirrored), forming a symmetric kaleidoscope-style enlargement.
- Category: tile-repeat
- Shape: constant 3x3 -> 6x6

### task143
- Rule: Multiple small shapes of different colors are scattered in the grid; one shape's color doesn't match the color conventionally used for its exact geometric form elsewhere, and it gets recolored to match that other shape's color.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task144
- Rule: A horizontal divider row of one color splits the grid into an upper and lower sub-pattern of different colors; the output is a small grid marking, with a new color, the cells where the two sub-patterns disagree.
- Category: other-complex
- Shape: constant 9x4 -> 4x4

### task145
- Rule: A key/cross-shaped divider line of one color partitions part of the grid into enclosed pockets; each enclosed pocket is flood-filled with a color depending on its relative size (smaller pocket gets one color, larger pocket gets another), while non-enclosed areas are left unchanged.
- Category: object-detect
- Shape: variable (in=out per example; sizes include 11x13, 11x16, 13x16, 15x16, 18x13)

### task146
- Rule: Three stacked 3x3 two-color pattern blocks are present; one block is selected based on a distinguishing property (its minority-color arrangement) and output unchanged as the 3x3 result.
- Category: object-detect
- Shape: constant 9x3 -> 3x3

### task147
- Rule: Finds all connected components made of one color and recolors the largest connected component to a new color, leaving smaller (isolated single-cell) components in their original color.
- Category: object-detect
- Shape: variable (in=out per example; sizes include 3x3, 4x4, 4x6, 5x5, 5x6)

### task148
- Rule: A vertical bar of one color has a horizontal ray drawn from its middle row toward the opposite side of the grid; if a lone marker dot lies in the ray's path it stops there and the dot is recolored, otherwise the ray reaches the far edge.
- Category: other-complex
- Shape: variable (in=out per example; sizes include 19x8, 20x10, 21x12)

### task149
- Rule: Divider lines split the grid into a 3x3 arrangement of panels, each containing scattered noise-colored dots; the output marks each panel with 1 if it contains more than one dot, otherwise 0.
- Category: counting
- Shape: constant 11x11 -> 3x3

### task150
- Rule: Mirrors the grid horizontally (left-right flip).
- Category: mirror-h
- Shape: variable (constant per example, always square in=out; sizes 3x3, 4x4, 6x6, 7x7)

### task151
- Rule: Finds the single "hole" cell inside the object made of the least-common color and recolors its 8 surrounding neighbor cells to color 4; everything else unchanged.
- Category: object-detect
- Shape: variable (per-example square, e.g. 4x4->4x4, 8x8->8x8, 6x6->6x6, 12x12->12x12)

### task152
- Rule: Builds a 2x2 kaleidoscope tiling: top-left quadrant is the input, top-right is the input mirrored left-right, and the bottom half is the entire top half mirrored top-to-bottom.
- Category: tile-repeat
- Shape: constant 3x3 -> 6x6

### task153
- Rule: Rotates the grid 270 degrees, finds the largest and smallest connected objects, fills a blank 3x3 canvas with the smallest object's color, paints the (normalized) largest object's shape onto it, then rotates the result back 90 degrees.
- Category: object-detect
- Shape: constant 10x10 -> 3x3

### task154
- Rule: Chooses whether to rotate the grid based on whether a color-2 object is portrait-shaped, then reorders/recolors color-5 objects relative to each other's horizontal center position.
- Category: other-complex
- Shape: constant 15x15 -> 15x15

### task155
- Rule: Flips the grid vertically (top-bottom mirror).
- Category: mirror-v
- Shape: variable (square per example, e.g. 5x5->5x5 or 7x7->7x7; size differs across examples)

### task156
- Rule: Finds all objects of color 4, then fills the "inbox backdrop" region of the smallest such object with color 1 and of the largest such object with color 2.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task157
- Rule: Crops the top band of the grid and, using the normalized positions of color-5 objects found elsewhere, reconstructs matching cells inside that cropped band as color 1.
- Category: other-complex
- Shape: constant 10x15 -> 10x15

### task158
- Rule: Finds the object with the most distinct colors, normalizes it, and stamps upscaled copies of its non-background-colored cells elsewhere on the grid at several scale factors.
- Category: object-detect
- Shape: variable

### task159
- Rule: Crops to the subgrid bounded by color-2 cells, takes the smallest object elsewhere in the grid, upscales it by a ratio derived from that subgrid's width, shifts it one cell, and paints it onto the cropped subgrid.
- Category: object-detect
- Shape: variable

### task160
- Rule: Finds all objects of exactly size 8, erases them, and stamps a small fixed diamond-shaped mark of color 2 anchored at each erased object's original upper-left corner.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task161
- Rule: Connects every pair of cells sharing the least-common color with straight lines, keeps only the resulting horizontal/vertical segments that fall outside the bounding box of the grid's non-background shapes, and recolors those kept segments while clearing the rest of the foreground.
- Category: other-complex
- Shape: variable

### task162
- Rule: Detects isolated 3x3 blocks that are entirely background (color 0) and marks the center cell of each such block with color 1.
- Category: object-detect
- Shape: constant 20x20 -> 20x20

### task163
- Rule: Finds the single color-4 cell and, based on which quadrant of the fixed-size grid its row/column falls into (thresholds at 3 and 7), paints a corresponding fixed diagonal pattern using colors 4 and 8.
- Category: other-complex
- Shape: constant 11x11 -> 11x11

### task164
- Rule: Concatenates the input with a left-right mirrored copy of itself to its right, doubling the width.
- Category: tile-repeat
- Shape: constant 3x3 -> 3x6

### task165
- Rule: Finds the largest object, then recolors other objects that sit in the same vertical band and below its top edge to the color of the remaining merged objects.
- Category: object-detect
- Shape: constant 20x20 -> 20x20

### task166
- Rule: Takes the first detected object and recolors its non-background cells to color 2 in place, leaving the rest of the grid unchanged.
- Category: object-detect
- Shape: variable

### task167
- Rule: Counts the number of distinct colors in the input. If there are 2 colors, draws the main diagonal in color 5 on a blank 3x3 canvas; if 3 colors, draws the anti-diagonal; if only 1 color, fills the top row with color 5.
- Category: counting
- Shape: constant 3x3 -> 3x3

### task168
- Rule: For each object, shoots a ray from a point offset by the object's "hole" position, in a direction chosen by comparing color counts among neighboring cells; a multi-step neighbor/color-count based line-drawing rule.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task169
- Rule: Groups objects by their pixel count and recolors all size-2 objects to color 3, size-3 objects to color 2, and size-4 objects to color 1.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task170
- Rule: Finds the largest and smallest objects' subgrids, downsamples the larger one by the width ratio between them, then fills the smaller subgrid's cells with background wherever the downsampled larger subgrid is background.
- Category: object-detect
- Shape: variable

### task171
- Rule: Recolors every cell on the outer border/edge of the (otherwise blank) grid to color 8, leaving interior cells unchanged.
- Category: recolor-conditional
- Shape: variable (e.g. 3x3, 4x3, 5x4, 7x6)

### task172
- Rule: Concatenates the input above a top-bottom mirrored copy of itself, doubling the height.
- Category: tile-repeat
- Shape: constant 3x3 -> 6x3

### task173
- Rule: Separates single-cell "noise" objects from the main pattern, then finds and recolors other occurrences of that main pattern's shape elsewhere in the grid.
- Category: other-complex
- Shape: variable

### task174
- Rule: Among all detected objects, finds the one whose cropped subgrid is symmetric under a left-right mirror and outputs that subgrid.
- Category: object-detect
- Shape: variable

### task175
- Rule: Overlays the grid with its own transpose (keeping the non-background color at each cell), replaces remaining background with the resulting most common color, then draws a diagonal ray from the origin using the color found at the origin cell.
- Category: symmetry-fill
- Shape: constant 21x21 -> 21x21

### task176
- Rule: Orders the background-colored (color-0) objects left-to-right and recolors every third one (positions that are multiples of 3) to color 4.
- Category: other-complex
- Shape: variable (height constant 3, width varies)

### task177
- Rule: Crops the grid to the single detected non-background object's bounding subgrid, then mirrors that subgrid left-right.
- Category: object-detect
- Shape: variable

### task178
- Rule: Checks whether the first row is degenerate (only one distinct value) to decide whether to transpose the grid first, then collapses the grid to a single row/column listing the sequence of each object's color ordered left-to-right.
- Category: other-complex
- Shape: variable

### task179
- Rule: Transposes the grid (swaps rows and columns, i.e. diagonal mirror).
- Category: transpose
- Shape: constant 3x3 -> 3x3

### task180
- Rule: Splits the grid into four same-size quadrants and overlays them together, keeping whichever quadrant has a non-background cell at each position, producing one merged quarter-size grid.
- Category: symmetry-fill
- Shape: constant 8x8 -> 4x4

### task181
- Rule: Checks the color at the corner of the color-4 region; depending on that value, shifts a mirrored copy of the color-8 region left or right by a fixed offset and fills those new positions with color 8.
- Category: other-complex
- Shape: constant 6x9 -> 6x9

### task182
- Rule: Finds the color-5 object whose shape exactly matches its own bounding-box outline, extracts the region inside that box, and re-paints its non-background cells back into the same box in a fixed relative orientation.
- Category: object-detect
- Shape: constant 20x20 -> 20x20

### task183
- Rule: Crops the subgrid marked by color 8, strips colors 8 and 1 from the remaining grid, compresses out empty rows/columns, upscales the result to half the color-8 subgrid's width, and re-stamps that subgrid's background-cell pattern onto it.
- Category: other-complex
- Shape: variable

### task184
- Rule: Finds every object larger than 2 cells and paints a single pixel of each such object's color at a compressed/downscaled position on a small output canvas representing each large object's relative location.
- Category: object-detect
- Shape: variable

### task185
- Rule: Removes the largest connected shape, crops to the bounding box of the remaining pieces, and recolors specific cells of background-colored sub-objects using the color derived from merging the grid's full-row/column "frontier" lines.
- Category: other-complex
- Shape: variable

### task186
- Rule: Counts the number of color-1 cells in the input and draws that many consecutive color-2 cells across the top row of a blank 3x3 canvas (starting from top-left); if the count is exactly 4, an extra center cell is also marked.
- Category: counting
- Shape: constant 3x3 -> 3x3

### task187
- Rule: Finds all connected objects and recolors every object that does NOT touch the grid border to color 2, while the remaining background becomes color 3.
- Category: object-detect
- Shape: variable

### task188
- Rule: If the grid is taller than it is wide (portrait), keep only the top half; otherwise (landscape or square) keep only the left half.
- Category: fixed-crop
- Shape: variable

### task189
- Rule: Crops to the region marked by color 3, strips colors 3 and 8 from the remainder, compresses out empty rows/columns, upscales the result by 3x, then re-stamps the background-cell pattern from the color-3 region onto it.
- Category: other-complex
- Shape: constant 9x9 -> 6x6

### task190
- Rule: Finds all size-1 "noise" pixels and the larger main object; for each noise pixel, draws a ray from the main object's center through that pixel's direction, recolored to the main object's color.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task191
- Rule: Locates a color-1 shape and color-4 marker cells; depending on which quadrant each occupies, rotates a cropped copy of the color-1 shape by 90/180/270 degrees and stamps it at the color-4 location(s).
- Category: other-complex
- Shape: constant 23x23 -> 23x23

### task192
- Rule: Removes the least-common color's cells (sets to background), then finds the new least-common remaining color and grows it into neighboring background cells that already have more than one same-colored neighbor; a flood-style thickening rule.
- Category: object-detect
- Shape: variable

### task193
- Rule: Finds cells of the least-common color whose immediate neighborhood contains more than 2 differing same-colored neighbors (thick interior cells) and erases just those cells to background, leaving thinner edge parts of that color.
- Category: object-detect
- Shape: variable

### task194
- Rule: Builds a 2x2 mosaic: top-left is the original grid, top-right is the grid rotated 90 degrees clockwise, bottom-left is the grid rotated 90 degrees counterclockwise, and bottom-right is the grid rotated 180 degrees.
- Category: tile-repeat
- Shape: constant 3x3 -> 6x6

### task195
- Rule: Crops to the first detected object's subgrid, builds a 2x3 tiled copy versus a 3x-upscaled copy, overlays them cellwise keeping matching cells, then downscales the result by 3 to find the object's repeating unit pattern.
- Category: object-detect
- Shape: variable

### task196
- Rule: Separates straight line-shaped objects (single row or column) from other shapes, keeps only the non-line objects whose outline equals their full shape (hollow boxes), and fills their matching cells with color 3.
- Category: object-detect
- Shape: variable

### task197
- Rule: Finds the largest object among all detected objects, determines which remaining piece sits at an extreme corner position, and recolors/swaps that piece using the other available palette color.
- Category: object-detect
- Shape: variable

### task198
- Rule: Finds all background-colored (color-0) "hole" objects; recolors the ones that are perfectly square in shape to color 3, and every other-shaped hole to color 4.
- Category: object-detect
- Shape: variable

### task199
- Rule: Takes the single detected object, shifts a copy of it one cell down, erases the original and paints the shifted copy in its place, then marks a horizontal band of color-4 cells positioned relative to the object's leftmost column.
- Category: object-detect
- Shape: variable

### task200
- Rule: Takes the single detected object's color and leftmost column position, then fills a periodic pattern of cells (spaced at fixed intervals derived from that position) with color 5 across the grid.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task201
- Rule: Locates a small frame marked by four corner cells of one color (color 4) with two colored guide-lines on its left/right interior edges, then reconstructs the frame's interior by pasting the matching-colored parts of a separate two-colored "key" shape (found elsewhere in the grid) into the left and right halves, matched by which color the guide-line indicates.
- Category: object-detect
- Shape: variable

### task202
- Rule: The grid is divided into horizontal or vertical solid-color stripes (orientation auto-detected); each stripe contains a few lone 0-colored noise cells, and each one is extended into a full line spanning the entire stripe at that same row/column.
- Category: other-complex
- Shape: variable (same-shape in/out per example, dims vary)

### task203
- Rule: The grid is a set of nested concentric square rings (like an onion/target pattern); the sequence of ring colors from outermost to innermost is reversed while the nested-square structure itself stays identical.
- Category: recolor-conditional
- Shape: variable (same-shape in/out per example, dims vary)

### task204
- Rule: Finds all square-shaped hollow-outline objects (color 1); recolors the interior of squares with an even side length to color 2, and squares with an odd side length to color 7, leaving non-square objects unchanged.
- Category: recolor-conditional
- Shape: variable (same-shape in/out per example, dims vary)

### task205
- Rule: Finds the largest object, crops to its bounding box, then keeps only the rows (and, after a 90-degree turn, columns) that contain more than 4 distinct colors, discarding the rest and reassembling the kept strips into a smaller composite grid.
- Category: other-complex
- Shape: variable

### task206
- Rule: There is a small multi-colored object and a separate lone marker cell (color 5) elsewhere; a copy of the object is pasted so that its own distinguishing corner cell ends up diagonally adjacent to the marker, while the original object is left in place.
- Category: object-detect
- Shape: variable (same-shape in/out per example, dims vary)

### task207
- Rule: Splits the grid into four quadrants (top-left, top-right, bottom-left, bottom-right) and outputs, for each cell position, whichever color is least common among the four overlaid quadrants.
- Category: symmetry-fill
- Shape: constant 5x5 -> 2x2

### task208
- Rule: Finds a small hollow-rectangle template object of the rarest color, then searches the rest of the noisy grid for every place where the same hollow-rectangle "hole" pattern (made of 0s) occurs and fills each matching occurrence's interior with that color.
- Category: object-detect
- Shape: constant 21x21 -> 21x21

### task209
- Rule: Extracts a colored sub-object's bounding box, strips out an internal marker-colored (4) dividing line, and rescales the remaining pattern based on a ratio of matched sub-shape widths, producing a resized crop.
- Category: other-complex
- Shape: variable

### task210
- Rule: Vertically concatenates the input with its own upside-down (row-reversed) copy placed below it, doubling the height into a top/bottom mirror-symmetric image.
- Category: tile-repeat
- Shape: constant 3x3 -> 6x3

### task211
- Rule: Builds a 3-row-by-2-column tiling of mirrored copies of the input (concatenating a horizontally-mirrored copy alongside the original, then stacking that block three times vertically) to form a larger symmetric pattern.
- Category: tile-repeat
- Shape: constant 3x2 -> 9x4

### task212
- Rule: Locates a horizontal marker line of color 5 and shoots diagonal rays outward from its row through the rest of the grid, adding new colored diagonal lines above and below the marker.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task213
- Rule: Removes the color-5 divider lines (transposing first if the dividers run vertically instead of horizontally), determines how many distinct non-5 colors are present, and builds a smaller n x n grid arranging those colors.
- Category: other-complex
- Shape: variable

### task214
- Rule: Takes the fixed 3x3 pattern in the top-left corner of a template row, rotates it 270 degrees (90 clockwise) and pastes it into the first empty bracketed box to the right, and rotates it 180 degrees and pastes that into the second empty box; everything else stays unchanged.
- Category: other-complex
- Shape: constant 3x11 -> 3x11

### task215
- Rule: Picks the first (topmost) object and repeats/tiles copies of it vertically at intervals equal to its own height, extending the periodic pattern both above and below its original position across the grid.
- Category: object-detect
- Shape: variable (same-shape in/out per example, dims vary)

### task216
- Rule: Finds all color-1 hollow-box objects and crops the grid down to the one whose enclosed empty interior area is largest.
- Category: object-detect
- Shape: variable

### task217
- Rule: Takes the small foreground dot pattern, then for every non-background cell of that pattern pastes a full copy of the same pattern into the corresponding block of an upscaled canvas (each cell "blown up" into a block containing the whole pattern or left empty), producing a self-similar fractal tiling.
- Category: tile-repeat
- Shape: constant 9x9 -> 9x9

### task218
- Rule: Crops to the bounding box of the large colored block pattern, then removes duplicate adjacent rows and columns (collapsing each uniform-color block down to a single cell), yielding a compact grid of the distinct color blocks.
- Category: other-complex
- Shape: variable

### task219
- Rule: Orders objects by topmost position, removes the topmost one, and lays down a trail of shifted/recolored copies below it whose sizes are derived from the other objects, forming a descending comet-like trail.
- Category: object-detect
- Shape: constant 15x10 -> 15x10

### task220
- Rule: Around every isolated single-cell marker, draws a 3x3 outline ring using a fixed color determined by the marker's own color (color 8 -> ring of 4, color 3 -> ring of 6, color 2 -> ring of 1), leaving the marker cell itself unchanged.
- Category: recolor-conditional
- Shape: variable (same-shape in/out per example, dims vary)

### task221
- Rule: Counts the number of background (0) cells in the input to get a number N, then tiles the input pattern repeatedly into an N x N grid of blocks, but only along a partial "L" region (first block-row fully tiled, subsequent rows only the first block filled), with the rest left blank.
- Category: counting
- Shape: variable (in constant 3x3, output size depends on background-cell count)

### task222
- Rule: Finds the single largest colored object in the grid and clears everything else to background, keeping only that object in its original position.
- Category: object-detect
- Shape: constant 16x16 -> 16x16

### task223
- Rule: Upscales the input by a factor of 3, replacing each cell with a 3x3 block of the same color (nearest-neighbor scale-up).
- Category: tile-repeat
- Shape: constant 3x3 -> 9x9

### task224
- Rule: Four lone marker cells of color 5 define the corners of a larger bounding rectangle; the color found inside a small existing hollow-rectangle template (elsewhere in the grid) is used to draw a new rectangle outline just inside that larger marker-defined bounding box.
- Category: object-detect
- Shape: variable (same-shape in/out per example, dims vary)

### task225
- Rule: A small 2x2 block of four distinct colors sits near the grid's center; each of the four outer diagonal quadrants gets filled entirely with the color that sits diagonally opposite it within that 2x2 block, creating a point-symmetric pinwheel expansion around the unchanged center block.
- Category: symmetry-fill
- Shape: constant 6x6 -> 6x6

### task226
- Rule: Grid lines of color 5 partition the grid into a block grid; fills the blocks lying along a diagonal path through that block grid (starting at the top-left block) with colors 1, 2, and 3 in order, leaving all other blocks unchanged.
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task227
- Rule: Splits the grid into top and bottom halves and marks, on a fresh blank canvas, the positions where both halves have a background (0) cell (i.e. the intersection of the two halves' zero-locations) with color 2.
- Category: symmetry-fill
- Shape: constant 8x4 -> 4x4

### task228
- Rule: Inside a hollow rectangle there are four corner-marker colors; each one is erased from its interior corner and re-drawn just outside the box at the corner diagonally opposite its original position (point symmetry through the box's center).
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task229
- Rule: Finds the single largest object in the grid and paints it onto a canvas filled entirely with color 5, discarding every other color/object.
- Category: object-detect
- Shape: constant 3x3 -> 3x3

### task230
- Rule: For every 2x2 (or similar) colored square object, marks the four cells diagonally just outside its corners with fixed colors 1 (top-left), 2 (top-right), 3 (bottom-left), and 4 (bottom-right).
- Category: object-detect
- Shape: variable (same-shape in/out per example, dims vary)

### task231
- Rule: Detects the repeating horizontal period of the pattern in the single non-background row and extends/tiles that periodic pattern to exactly double the original width, keeping other rows unchanged.
- Category: tile-repeat
- Shape: variable (width always doubles)

### task232
- Rule: From each object's center, shoots a horizontal ray of the object's own color rightward to the edge of the grid, and additionally marks a fixed set of reference positions along one edge with color 5.
- Category: object-detect
- Shape: variable (same-shape in/out per example, dims vary)

### task233
- Rule: A large framed grid contains internal "holes"/gaps, and several small separate patches elsewhere in the image are matched and inserted into the holes whose shape they fit, then the result is cropped to the frame's bounding box.
- Category: object-detect
- Shape: variable

### task234
- Rule: Identifies which foreground object is "hollow" (has an internal hole) versus "solid", then recolors any cell of the solid object that has more than 3 neighboring cells of the hollow object's color to that same color (a local growth/contagion effect).
- Category: object-detect
- Shape: variable (same-shape in/out per example, dims vary)

### task235
- Rule: The grid contains three 4x4 glyph-like sub-patterns (made of color 5 and background) separated by blank columns; each glyph is matched against a fixed set of known shapes and classified, and the output is a 3x3 grid whose rows are filled with the corresponding matched colors.
- Category: other-complex
- Shape: constant 4x14 -> 3x3

### task236
- Rule: Splits the grid into top and bottom halves and marks, on a fresh blank canvas, the positions where exactly one of the two halves (not both) has a background (0) cell, using color 3.
- Category: symmetry-fill
- Shape: constant 9x4 -> 4x4

### task237
- Rule: From each object's center, shoots a horizontal ray of the object's own color rightward across the grid, and adds an extra single marker cell of color 0 at a fixed offset relative to the grid's overall size.
- Category: object-detect
- Shape: variable

### task238
- Rule: Identifies the object with the fewest distinct colors and the "other" object, aligns/crops the grid based on the best-matching offset between the two shapes, and outputs the resulting cropped composite.
- Category: object-detect
- Shape: variable

### task239
- Rule: Ranks the objects by size and renders a bar-chart-like column for each one: a solid block of the object's own color whose height reflects its relative size, stacked above filler background rows.
- Category: counting
- Shape: variable

### task240
- Rule: Tests the 4 rotations (0/90/180/270) of the grid, picks whichever's top-left quadrant has the most distinct colors, then combines that rotation with its own horizontal mirror by taking the cell-wise maximum color, producing a symmetrized grid.
- Category: symmetry-fill
- Shape: constant 19x19 -> 19x19

### task241
- Rule: Transposes the grid (flips it across the main diagonal, swapping rows and columns).
- Category: transpose
- Shape: variable (each example is square, in=out, but the side length varies across examples)

### task242
- Rule: Horizontally mirrors the whole grid, then crops the mirrored result down to the bounding box of wherever the original (unmirrored) grid had background-colored (0) cells.
- Category: object-detect
- Shape: constant 16x16 -> 3x3

### task243
- Rule: Finds every background-colored (0) blob that is adjacent to any color-1 cell and recolors that entire blob to color 1, leaving background regions not touching a 1 unchanged.
- Category: recolor-conditional
- Shape: variable (same-shape in/out per example, dims vary)

### task244
- Rule: Removes duplicate uniform rows/columns (compresses the grid), mirrors it horizontally, then downscales it by a factor equal to the smallest object's width, producing a small crop that encodes which sub-cell of a grid-of-cells contains the marker color.
- Category: other-complex
- Shape: variable

### task245
- Rule: A colored shape and a set of marker cells (color 3) are present; the shape is translated so that its own bounding-box corner sits one cell diagonally inside the marker cells' bounding-box corner, i.e. it "snaps" to align with the markers.
- Category: object-detect
- Shape: variable (same-shape in/out per example, dims vary)

### task246
- Rule: Two lone single-cell markers (colors 2 and 3) sit in an otherwise empty grid; draws an orthogonal right-angle connecting path (elbow line) of color 8 between them, stopping one cell short of each marker.
- Category: object-detect
- Shape: variable (same-shape in/out per example, dims vary)

### task247
- Rule: Finds the object(s) with the maximum size, orders them left-to-right, and builds a small grid where each qualifying object's color fills its own full column, height equal to that maximum size, then transposes the result.
- Category: object-detect
- Shape: variable

### task248
- Rule: A single marker cell sits in the bottom row; a zigzag/bouncing trajectory is traced upward row by row, moving one column at a time and reflecting off the left/right walls like a bouncing ball, marking the visited cell in each row with color 1.
- Category: other-complex
- Shape: variable (same-shape in/out per example, dims vary)

### task249
- Rule: Horizontally concatenates the input with an identical copy of itself, doubling the width.
- Category: tile-repeat
- Shape: variable (width always doubles)

### task250
- Rule: Several lone marker cells of color 5 are each moved to the nearest point touching a separate 2x2 colored block object (snapping each marker onto the object's boundary at its closest approach), clearing their original positions.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task251
- Rule: Finds a rectangular outline drawn in one color and flood-fills its enclosed interior with a second color, leaving the outline and background untouched.
- Category: object-detect
- Shape: variable (constant square per example, e.g. 8x8 -> 8x8, 12x12 -> 12x12, but size differs across examples)

### task252
- Rule: For each diagonal line segment of a single color, recolors alternating pixels along the line to a second fixed color (starting unchanged at one end), producing a dashed-line effect.
- Category: other-complex
- Shape: variable (constant square per example, e.g. 3x3 -> 3x3, 8x8 -> 8x8)

### task253
- Rule: Four small L-shaped "corner" pieces are scattered around a 13x13 grid; the rule identifies each piece's corner orientation and assembles them into their matching corners of a 4x4 output square.
- Category: object-detect
- Shape: constant 13x13 -> 4x4

### task254
- Rule: The grid contains several vertical bars (columns) of a marker color with different heights; recolors the tallest bar to one color, the shortest bar to a second color, and removes (blanks) all other bars.
- Category: counting
- Shape: constant 9x9 -> 9x9

### task255
- Rule: Finds the largest all-background (empty) rectangular region hidden among scattered noise cells and fills it with a solid marker color.
- Category: object-detect
- Shape: constant 30x30 -> 30x30

### task256
- Rule: Detects a horizontal segment of one color and draws a triangular wedge above it (width = segment length + row distance, in one color) and a shrinking triangular wedge below it (width = segment length - row distance, in a second color), anchored at the segment's start column.
- Category: object-detect
- Shape: variable

### task257
- Rule: The grid is split into four quadrants by a cross of a divider color, each quadrant containing sparse marks of its own color; the quadrants are overlaid into one 4x4 grid, with a fixed color-priority order resolving any cell where multiple quadrants have a mark.
- Category: other-complex
- Shape: constant 9x9 -> 4x4

### task258
- Rule: Pairs of same-colored dots sit two cells apart (horizontally or otherwise) in an otherwise empty grid; fills the single empty cell exactly between each such pair with a second color.
- Category: object-detect
- Shape: variable

### task259
- Rule: Crops the grid to the bounding box of all non-background-colored cells, replacing the background color inside that crop with 0.
- Category: object-detect
- Shape: variable

### task260
- Rule: A diagonal line of one color runs through the grid with a small triangular patch of a second color abutting it; the rule removes the patch and adds a second diagonal line parallel to the first, offset by an amount determined by the patch's size/shape.
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task261
- Rule: Shifts the entire grid down by one row (dropping the last row and inserting a blank row at the top) and recolors all cells of the shape's original color to a second fixed color.
- Category: other-complex
- Shape: variable (constant square per example)

### task262
- Rule: Each row contains a single marker cell in one of 3 columns; fills the entire row with a color determined solely by which column the marker occupies (a fixed column-to-color mapping), regardless of row index.
- Category: recolor-conditional
- Shape: constant 3x3 -> 3x3

### task263
- Rule: The input is a strip of several equal-size sub-blocks stacked/placed side by side, each in its own color; most share the same internal cell-pattern but one is different (the "odd one out"); the output is that unique block, cropped.
- Category: object-detect
- Shape: variable input -> constant 3x3 output

### task264
- Rule: Several 3x3 tiles (mostly one fill color with a small colored "notch" in one corner) are scattered across the grid; the rule reads each notch's corner orientation and reassembles the tiles into their corresponding positions of a 9x9 grid.
- Category: object-detect
- Shape: variable input -> constant 9x9 output

### task265
- Rule: Finds the largest all-background rectangular region hidden among scattered noise cells and fills it with a solid marker color (same style of task as task255).
- Category: object-detect
- Shape: constant 18x18 -> 18x18

### task266
- Rule: A single marker cell sits somewhere in the grid; recolors its four diagonal neighbors using a fixed per-direction color mapping (up-left, up-right, down-left, down-right each get their own fixed color), clipped at grid edges, and removes the original marker.
- Category: object-detect
- Shape: constant 3x5 -> 3x5

### task267
- Rule: A connected blob of one color and a single isolated pixel of a different color both appear in the grid; recolors the entire blob to the isolated pixel's color and removes the isolated pixel.
- Category: object-detect
- Shape: constant 7x7 -> 7x7

### task268
- Rule: A container-like shape has a gap/opening on one side; draws rays of a second color spreading outward (like light through a slit) from the gap toward the edge of the grid.
- Category: other-complex
- Shape: variable (constant square per example)

### task269
- Rule: Counts the number of distinct non-background colors present in the 3x3 input, then upsamples (nearest-neighbor / kron) the whole input by that count as the scale factor.
- Category: counting
- Shape: variable (input always 3x3, output NxN where N = 2,3,4,5 depending on example)

### task270
- Rule: A unique "center" marker and several far-away satellite cells of a matching companion color lie in the four cardinal directions from the center; redraws each satellite immediately adjacent to the center (distance 1) in the direction it originally occupied, removing the far originals.
- Category: object-detect
- Shape: constant 15x15 -> 15x15

### task271
- Rule: Several 3x3 sub-blocks are embedded in the 9x9 grid; outputs the one 3x3 block containing the most cells of a particular marker color.
- Category: counting
- Shape: constant 9x9 -> 3x3

### task272
- Rule: Connected components of one color are examined; isolated single cells (no same-color neighbor) are recolored to a second color, while larger connected groups keep the original color.
- Category: object-detect
- Shape: variable

### task273
- Rule: Four dots of one color mark the corners of a small rectangle; fills the interior cell(s) of that rectangle with a second color.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task274
- Rule: A "container" shape made of one color is partially filled with a second color representing a liquid/fill level; the fill level is measured and rendered as a compressed 3x3 bar-meter, filled proportionally from the top.
- Category: counting
- Shape: variable input -> constant 3x3 output

### task275
- Rule: The input is split into a left half (a small NxN grid of sparse colored cells) and a right half (an NxN "stamp" template shown in one marker color); the output upsamples the left grid by stamping each nonzero cell with the template shape (kron-style expansion), in that cell's color.
- Category: other-complex
- Shape: variable

### task276
- Rule: Recolors every cell of one fixed color (6) to a second fixed color (2); all other cells are left unchanged.
- Category: recolor-map
- Shape: variable

### task277
- Rule: Connected shapes are classified by form: solid rectangular blocks are recolored to one color, hollow/"C"-shaped frames are recolored to a different color.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task278
- Rule: Several scattered dots of one color exist; finds the one pair of dots that lies closest together (adjacent) and draws a rectangular border of a second color around their bounding box, leaving other dots untouched.
- Category: object-detect
- Shape: variable

### task279
- Rule: Shapes made of a marker color are examined for topology; closed loop/rectangle outlines are recolored to a second color, while open/branching (plus-shaped) figures are left unchanged.
- Category: object-detect
- Shape: constant 9x11 -> 9x11

### task280
- Rule: Rectangular blocks of one color each contain a single "flawed" cell of a second color on one edge, indicating a direction; a 3-line-wide ray (center line in the flaw color, flanking lines in the block color) shoots outward from that edge to the grid boundary.
- Category: object-detect
- Shape: variable (constant per example, 10x10)

### task281
- Rule: A framed box (border color + interior color) and a single distant marker dot exist; the box is stretched/extended so its border reaches the row of the marker, and the marker is absorbed into the extension.
- Category: object-detect
- Shape: variable

### task282
- Rule: Each isolated dot of one color is expanded into a fixed ring stamp: its four diagonal neighbors become one color, its four orthogonal neighbors become a second color, and the original center cell is cleared.
- Category: object-detect
- Shape: constant 9x9 -> 9x9

### task283
- Rule: Solid square/rectangular blocks of one color are recolored by depth from their edge: corner cells get one color, remaining border cells get a second color, and the interior gets a third color.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task284
- Rule: Two marker dots of different colors sit on the same row some distance apart; the rule draws a connecting line between them plus a small diamond-shaped flourish at each end, sized in relation to their separation.
- Category: object-detect
- Shape: variable

### task285
- Rule: Several shapes each have a small companion "seed" cell of a different color nearby; the rule duplicates/copies the main shape in the seed's color at a location and offset determined by that seed.
- Category: object-detect
- Shape: variable

### task286
- Rule: A maze-like corridor network is drawn with wall cells of one color; starting from a short seed pattern of alternating colors placed in one corridor, the rule extends the alternating color fill through the entire connected corridor path.
- Category: other-complex
- Shape: variable

### task287
- Rule: The grid has full 4-fold (horizontal, vertical, 180-rotation, diagonal) symmetry, but some cells are occluded with a marker color; fills the occluded cells using the value found at their symmetric counterpart position.
- Category: symmetry-fill
- Shape: constant 16x16 -> 16x16

### task288
- Rule: A small cross/plus-shaped figure has one arm colored differently from the rest (a marker color); reflects that marker color onto the mirror-image position across the figure's center to complete a symmetric figure.
- Category: symmetry-fill
- Shape: variable

### task289
- Rule: Counts the number of distinct non-background colors present in the 3x3 input, then upsamples (nearest-neighbor / kron) the whole input by that count as the scale factor (same mechanism as task269).
- Category: counting
- Shape: variable (input always 3x3, output NxN where N = 2,3,4 depending on example)

### task290
- Rule: Crops the grid to the bounding box of the single object present (a bordered rectangle with a distinct interior color) and swaps the border color with the interior color.
- Category: object-detect
- Shape: variable

### task291
- Rule: Several solid-colored rectangular objects appear in the grid, all but one being perfect filled rectangles; outputs the single color (as a 1x1 grid) belonging to the one object whose shape is irregular/hollow (not a solid rectangle).
- Category: object-detect
- Shape: variable input -> constant 1x1 output

### task292
- Rule: A repeating checkerboard-like zig-zag pattern of one color spans the grid; recolors the crossing cells at every third column (a fixed periodic interval) to a second color.
- Category: recolor-conditional
- Shape: variable (constant 3 rows, width varies)

### task293
- Rule: A vertical bar and a horizontal bar of different colors and thicknesses cross each other; at the intersection, the thicker bar's color is drawn on top (replacing the thinner bar's color there).
- Category: object-detect
- Shape: variable

### task294
- Rule: Solid rectangular blocks of one color have their outer one-cell-thick border left unchanged while the interior (excluding the border) is recolored to a second color.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task295
- Rule: Given a single row of width W, repeats that row vertically W/2 times to build a taller output of the same width.
- Category: tile-repeat
- Shape: variable (input 1xW -> output (W/2)xW)

### task296
- Rule: The grid is divided into four quadrants by blank divider rows/columns, each quadrant holding a mirrored copy of the same sparse pattern; the quadrants are folded/overlaid together (keeping cells where the mirrored copies agree) into a compact 3x3 grid.
- Category: symmetry-fill
- Shape: constant 5x7 -> 3x3

### task297
- Rule: The first two rows hold a short color "key" (e.g. two colors) and a separator row; the rule continues filling the remaining rows below, each solid-filled with the next color in the repeating key sequence.
- Category: other-complex
- Shape: variable

### task298
- Rule: The grid is made of concentric square rings of different colors; cyclically shifts every ring's color inward by one layer, with the innermost color wrapping around to become the new outermost ring.
- Category: object-detect
- Shape: variable (constant square per example, e.g. 6x6 -> 6x6, 8x8 -> 8x8)

### task299
- Rule: A short vertical line segment of one color and a short horizontal line segment of a second color both appear; extends each segment to span the full grid (full row / full column) and marks their intersection cell with a third color.
- Category: object-detect
- Shape: constant 6x6 -> 6x6

### task300
- Rule: Several differently-colored, differently-shaped objects are scattered in the grid; finds the object with the most cells (the largest one) and crops the output to its bounding box.
- Category: object-detect
- Shape: variable

### task301
- Rule: Each row contains a contiguous horizontal bar of a single non-background color; each such bar is pushed (right-aligned) to the right edge of its row, then all rows (bars plus empty rows) are vertically re-stacked in ascending order of bar length, empty rows moved to the top and the longest bar placed at the bottom.
- Category: other-complex
- Shape: variable (input/output always match each other's H x W per example, but dims differ across examples: e.g. 10x7, 7x4, 3x3, 11x8)

### task302
- Rule: Grid contains closed rectangular frames (colored borders) enclosing a hollow interior of background 0s; each interior hole is filled with a color determined by the hole's cell-count (1 cell -> 6, 4 cells -> 7, 9 cells -> 8), leaving the frame borders and everything else unchanged.
- Category: counting
- Shape: constant 12x12 -> 12x12

### task303
- Rule: The grid contains exactly one row and one column composed entirely of background 0s (acting as divider lines); every cell belonging to that all-zero row or all-zero column is recolored to 2, while all other cells (including scattered isolated 0s elsewhere) are left unchanged.
- Category: object-detect
- Shape: variable (e.g. 12x19, 12x14, 17x15, test 27x25)

### task304
- Rule: Input is a 3x3 grid; the most frequent (majority) color's cell positions are used as a mask over a 3x3 tiling of the output - for each cell equal to the majority color, a full copy of the input is placed in the corresponding block of the 9x9 output, other blocks are left as 0 (a fractal/self-similar replication keyed on the majority color).
- Category: counting
- Shape: constant 3x3 -> 9x9

### task305
- Rule: The grid holds a repeating diagonal wrap-around numeric sequence (e.g. 1,2,3,4,5 repeating along diagonals) that has been corrupted at scattered cells with noise values (including 0); the output restores every corrupted cell to the value dictated by the underlying periodic diagonal pattern.
- Category: symmetry-fill
- Shape: constant 16x16 -> 16x16

### task306
- Rule: The grid is split by one or two full straight lines of a marker color (e.g. a solid row and/or solid column of color 4) into quadrants; one quadrant/half contains the "real" content and the others are blank (0s) in the input, and the output copies that populated region into the blank region(s) on the other side of the divider line(s).
- Category: symmetry-fill
- Shape: variable (e.g. 19x19, 19x9, test 19x29)

### task307
- Rule: Scales the entire input up by a factor of 2, replacing each input cell with a solid 2x2 block of the same color (nearest-neighbor upscale) - pure geometric scaling with no color-dependent logic.
- Category: tile-repeat
- Shape: variable absolute sizes but constant 2x scale ratio (e.g. 3x3 -> 6x6, 2x2 -> 4x4, 5x5 -> 10x10)

### task308
- Rule: A large grid filled mostly with one fill color contains several small sparse stamped patterns of distinct marker digits scattered at different locations; the output extracts/merges these scattered marker patterns into one small dense grid (looks like an overlay of multiple stamped cross/motif patterns found by locating non-fill-color cells).
- Category: object-detect
- Shape: variable (e.g. 12x11 -> 5x5, 10x8 -> 3x3, 12x14 -> 5x5)

### task309
- Rule: Recolors every cell of color 7 to color 5; all other colors (e.g. 1, 8) are left unchanged.
- Category: recolor-map
- Shape: variable (height constant at 3, width varies: 3x6, 3x4, 3x5, etc.)

### task310
- Rule: A large grid is tiled with repeating colored blocks separated by mixed-color divider lines; the output crops out a single distinguishing/anomalous block (identified via a divider row or column whose color composition differs from the rest) and frames it with a border of the distinguishing color.
- Category: object-detect
- Shape: variable (e.g. 24x24 -> 7x7, 26x26 -> 7x7, test 24x24 -> 6x6; output size itself is not constant)

### task311
- Rule: Horizontally concatenates the input with its left-right mirror image (output = input side-by-side with its horizontal flip).
- Category: other-complex
- Shape: constant 3x3 -> 3x6

### task312
- Rule: Each connected shape/region contains some cells of a "filler" color (5) alongside a small number of cells showing the true marker color; every 5-cell is recolored to match the true color of the shape/region it belongs to (color propagation via connected components), true-colored cells are left unchanged.
- Category: object-detect
- Shape: constant 12x12 -> 12x12

### task313
- Rule: The top-left portion of the grid contains a periodic two-color checkerboard/striped pattern next to a solid "filler" color block; the whole output grid is filled by continuing/tiling that detected periodic pattern across the entire canvas, replacing the filler region.
- Category: symmetry-fill
- Shape: variable (train/test seen: 6x6, 8x8, 11x11, 18x18; arc-gen has 16 distinct shape pairs, all square input=output)

### task314
- Rule: Finds isolated single-cell (size-1) colored objects; for each pair of same-color single cells that share a row or column, draws a straight line of that color connecting them, leaving everything else unchanged.
- Category: object-detect
- Shape: constant 8x8 -> 8x8

### task315
- Rule: Tiles the 3x3 input into a 9x9 grid (3x3 copies of itself), but a tile-block is only filled in with the tiled content when the corresponding original input cell has color 2; all other blocks are left as zero (color-conditional fractal upscale).
- Category: other-complex
- Shape: constant 3x3 -> 9x9

### task316
- Rule: Detects scattered single-pixel colored objects on a background, orders them by position (leftmost first), and packs their colors into a small 3x3 (or similar) output grid, mirrored/transformed per a fixed layout rule.
- Category: object-detect
- Shape: constant 10x10 -> 3x3

### task317
- Rule: For every isolated non-background pixel, draws a solid 3x3 box (color 1) centered on that pixel (the pixel's outbox/backdrop), overlapping boxes merge; original pixels are overwritten.
- Category: object-detect
- Shape: constant 9x9 -> 9x9

### task318
- Rule: Splits the input into a top 4-row half and a bottom 4-row half (separated by a row of filler color 4); output is a canvas of color 3 with 0 placed wherever both halves have a 0 at the same position (logical AND of the two halves' zero-masks).
- Category: other-complex
- Shape: constant 9x4 -> 4x4

### task319
- Rule: Grid contains multiple distinct colored shapes on a background; the largest shape is removed, and among the remaining shapes the one whose normalized/rescaled outline best matches the removed largest shape's outline is selected and output as a small cropped grid.
- Category: object-detect
- Shape: variable (train/test/arc-gen show many distinct input/output shape pairs, e.g. 17x17->5x5, 18x18->5x3, 15x17->3x5; 139 distinct shape-pairs in arc-gen)

### task320
- Rule: Grid contains a staircase/zigzag object made of color 2 on background 0; for each such object, a line is drawn from its upper-left corner to its center of mass, and the cells on that connecting path are recolored from 2 to 8.
- Category: object-detect
- Shape: variable (width constant at 9, height varies 7/8/9/11; input and output always share the same shape per example)

### task321
- Rule: Input is split by full separator columns of one color into 3 equal-width panels; output overlays the panels cell-by-cell, taking the value from the first panel (in left-to-right order) that is non-zero at that cell, else 0.
- Category: other-complex
- Shape: constant 4x14 -> 4x4

### task322
- Rule: For each column, finds the single non-zero cell and fills every cell below it (same column, same value) down to the bottom row, leaving all cells above it unchanged (0).
- Category: other-complex
- Shape: constant 3x3 -> 3x3

### task323
- Rule: Grid contains one marker pixel (color 8) on a background of 0s; output draws an expanding zigzag/staircase pattern of color 5 emanating from the marker (diagonally in both directions), while keeping the marker pixel itself.
- Category: object-detect
- Shape: constant 13x13 -> 13x13

### task324
- Rule: Grid is divided into colored background regions (e.g. two alternating fill colors) with several isolated marker pixels of other colors scattered inside; output redraws diagonal ray/reflection patterns radiating from each marker across the regions while leaving the rest of the background intact. Same size as input.
- Category: other-complex
- Shape: variable (input size differs per example; output always matches its own input size)

### task325
- Rule: Counts the number of separate non-background blobs (color 8) in the input, then outputs an NxN grid (N = blob count) containing a diagonal identity-like pattern (8s on the main diagonal, 0 elsewhere).
- Category: counting
- Shape: variable (input size varies; output is NxN where N = number of blobs, so output size also varies)

### task326
- Rule: Crops the input to its fixed top-left 2x2 corner block, discarding everything else (input may be a tiled/repeating pattern but only the top-left 2x2 is kept).
- Category: fixed-crop
- Shape: variable input -> constant 2x2 output

### task327
- Rule: Takes the NxN input and repeatedly stamps it onto a 2N x 2N canvas of zeros, shifted diagonally down-right by one cell at each step (offsets 0..2N-1, clipped at the edges), with each stamp only overwriting cells where its value is non-zero (later/lower-right stamps take precedence over earlier ones).
- Category: other-complex
- Shape: constant 3x3 -> 6x6

### task328
- Rule: Input has 1-2 isolated marker pixels near opposite corners/edges on a background of 0s; output fills in elaborate diagonal staircase/ray patterns using colors derived from the markers, radiating from each marker across the grid, same size as input.
- Category: other-complex
- Shape: variable (input size differs per example; output always matches its own input size)

### task329
- Rule: Zeroes out every column except the single middle column (width//2), which is copied unchanged from the input; all other columns become all-zero.
- Category: other-complex
- Shape: constant per example (always square, e.g. 3x3, 5x5, 7x7, 9x9 -> same size), but square size varies across examples

### task330
- Rule: Finds connected non-background blobs; recolors every blob whose cell count is exactly 6 to color 2, and recolors every other non-background blob (any other size) to color 1.
- Category: counting
- Shape: constant 10x10 -> 10x10

### task331
- Rule: For each isolated pixel of color 1, colors its four orthogonal neighbor cells with fixed colors (up=2, down=8, left=7, right=6); all other cells unchanged.
- Category: recolor-conditional
- Shape: constant 10x10 -> 10x10

### task332
- Rule: Conceptually mirrors the grid left-right, recolors color-5 cells sitting in even columns of that mirrored view to color 3 (a column-parity-dependent subset of the 5s), then mirrors back; remaining cells unchanged.
- Category: recolor-conditional
- Shape: variable (in=out per example; height always 3, width varies e.g. 3x10, 3x12, 3x17)

### task333
- Rule: Finds single-cell dots that are row- or column-aligned with a small 2x2 colored block, then draws a straight line of the dot's own color connecting it to the block.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task334
- Rule: Determines which single foreground color is present in the input (1, 2, or other) and, based purely on that color identity, draws a plus/cross of color 5 through a specific anchor cell on a fresh blank 3x3 canvas; the actual input pixel positions/shapes are ignored.
- Category: other-complex
- Shape: constant 5x5 -> 3x3

### task335
- Rule: Locates one cell of color 8 and one cell of color 2, then draws a right-angle (L-shaped) path of color 4 connecting them via the corner point formed by intersecting their row and column, overwriting only background cells.
- Category: object-detect
- Shape: variable (in=out per example; dims vary widely, e.g. 10x12, 8x11, 12x13)

### task336
- Rule: Finds a square/rectangular ring-shaped object of color 5, fills its interior with color 8, and shoots an additional ray of color 8 outward from the single gap/notch in the ring's border in the direction the notch faces.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task337
- Rule: Swaps every cell of color 5 with color 8 and vice versa everywhere in the grid; all other colors are left unchanged.
- Category: recolor-map
- Shape: variable (in=out per example; dims vary, e.g. 5x5, 3x3, 4x4)

### task338
- Rule: Finds all non-square (hollow/irregular-shaped) colored objects, fills their interior bounding area with color 3, and zeroes out all color-2 cells; square-shaped objects are left untouched.
- Category: object-detect
- Shape: variable (in=out per example; dims vary, e.g. 10x10, 25x25)

### task339
- Rule: Counts the number of cells of the single non-background color present in the input, and outputs a single-row grid of that many cells, all filled with that color.
- Category: counting
- Shape: input constant 3x3, output variable width (e.g. 1x1, 1x2, 1x3, 1x5, up to 1x9)

### task340
- Rule: Finds isolated single-pixel dots whose color matches some larger multi-cell object elsewhere in the grid, then draws a straight trail of the dot's color connecting/moving each dot toward its color-matching object.
- Category: object-detect
- Shape: variable (in=out per example; dims vary, e.g. 10x15, 12x12, 14x17)

### task341
- Rule: Two rectangular blocks of different colors sit apart from each other; the gap region directly between them (aligned to their overlap) is filled in with color 8, everything else unchanged.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task342
- Rule: A central 2x2 block plus four scattered single colored pixels; each scattered pixel's color is written into whichever corner of the 2x2 block is nearest to it (Manhattan distance), and everything else (original scattered pixels, block color) is erased to 0.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task343
- Rule: Each row contains a short pattern already repeated twice near the left edge; that periodic unit is continued/tiled rightward to fill the rest of the row, leaving all-zero rows untouched.
- Category: tile-repeat
- Shape: constant 5x15 -> 5x15

### task344
- Rule: Wherever a cell of color 3 is orthogonally adjacent to a cell of color 2, the 3 becomes 8 and the adjacent 2 becomes 0; all other cells (including non-adjacent 2s) are left unchanged.
- Category: recolor-conditional
- Shape: variable (input=output same shape per example, but shape differs across examples)

### task345
- Rule: The bottom row encodes a set of marker columns (color 2) which project upward as vertical stripes filling every row above; wherever a special isolated "5" pixel occurs elsewhere in the grid, the stripe passing near it is deflected sideways by one column from that row upward (the 5 pixel itself is preserved).
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task346
- Rule: The grid has scattered pixels of two non-background colors; the output is a single pixel equal to whichever color is broken into more separate small connected components (the "noisier"/more scattered color) rather than forming one compact solid shape.
- Category: counting
- Shape: variable input size -> constant 1x1 output

### task347
- Rule: Splits the input into left and right halves of equal width, then unions them: outputs color 6 wherever either half has a non-zero cell at that position, and 0 where both halves are zero there.
- Category: symmetry-fill
- Shape: constant 3x6 -> 3x3

### task348
- Rule: A short vertical line segment has its top end sprout a growing diamond/triangular ripple pattern extending upward, alternating between the line's color and 8 based on distance from the line's top; cells at/below the line are unchanged.
- Category: other-complex
- Shape: variable (input=output same shape per example, but shape differs across examples)

### task349
- Rule: Each small 2x2 colored block acts as a seed; concentric square (Chebyshev-distance) rings are grown outward from each seed and colored alternately (e.g. color 3 then color 1), filling the whole grid, with rings from different seeds merging where they meet.
- Category: other-complex
- Shape: variable (always square, input=output same size per example, but size differs across examples, e.g. 10x10 up to 30x30)

### task350
- Rule: Isolated single pixels of the same color that share a row or column get connected by a straight line of color 8 drawn between them (original endpoint pixels keep their color); pixels with no aligned same-color partner are left unchanged.
- Category: object-detect
- Shape: variable (input=output same shape per example, but shape differs across examples)

### task351
- Rule: Input is a 16x16 grid with 4-fold mirror symmetry (kaleidoscope-like pattern) plus scattered single-cell color noise; output is a small 5x5 "clean" motif recovered from the symmetric/repeating structure.
- Category: symmetry-fill
- Shape: constant 16x16 -> 5x5

### task352
- Rule: Draws a solid halo/outline (color 1) directly surrounding each isolated pixel of a specific marker color, leaving the original dot and background untouched; other marker colors present get no halo.
- Category: object-detect
- Shape: variable

### task353
- Rule: A single pair of markers (colors 3 and 4) sit apart on an otherwise blank grid; the color-3 marker moves one step diagonally/directly toward the fixed color-4 marker.
- Category: object-detect
- Shape: variable

### task354
- Rule: A row of single-cell color markers acts as a legend; solid rectangular color-5 blocks elsewhere get recolored into repeating striped bands using the legend colors keyed by column position.
- Category: recolor-conditional
- Shape: constant 10x10 -> 10x10

### task355
- Rule: Grid is split into large color-block regions, each mostly one solid color with a handful of tiny same-colored anomaly pixels sprinkled in; output is the single color value that is rarest/least frequent among those anomalies across the whole grid.
- Category: counting
- Shape: variable

### task356
- Rule: Grid contains several isolated single-pixel dots of one color; every pair of dots sharing a row or column gets connected by a straight line of that color (chained through shared rows/columns), leaving unconnected dots untouched.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task357
- Rule: A single seed pixel sits in the last row of an otherwise blank grid; output fills the whole grid with a diagonal alternating two-color stripe wave emanating from the seed's column.
- Category: other-complex
- Shape: variable

### task358
- Rule: A tiny periodic seed pattern sits near one edge of an otherwise empty grid; output extends/tiles that detected periodic pattern outward to fill the entire grid.
- Category: other-complex
- Shape: variable

### task359
- Rule: Grid is divided into contiguous bands (rows or columns) each dominated by one background color plus scattered noise pixels; output replaces every cell in each band with that band's dominant color, erasing the noise.
- Category: recolor-conditional
- Shape: variable

### task360
- Rule: A vertical marker line (uniform color) splits the grid into a left half and right half holding complementary sparse shapes; output drops the marker line and overlays the two halves (right half mirrored) keeping whichever cell is non-background, producing one half-width result. Verified programmatically to fold-and-overlay correctly.
- Category: symmetry-fill
- Shape: constant 10x9 -> 10x4

### task361
- Rule: A small colored cross/diamond object near the grid center shows only part of a symmetric shape; output completes it into a full point-symmetric diamond outline by mirroring the existing partial pattern outward.
- Category: symmetry-fill
- Shape: constant 10x10 -> 10x10

### task362
- Rule: A vertical line and a full horizontal line of one color intersect, plus a few stray same-colored dots near one end of the vertical line; the vertical line shifts left by an amount equal to the count of the stray dots, and the stray dots are removed.
- Category: counting
- Shape: constant 10x10 -> 10x10

### task363
- Rule: Sparse grid of background 0 plus colors 2 and 5; extra color-2 pixels are inserted adjacent to certain existing 5's, apparently completing a fine-grained repeating local motif (exact placement rule not fully pinned down).
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task364
- Rule: Grid contains several small objects made of one color in different shapes (bars, crosses, corners, single dots); each object is recolored to one of several new colors depending on its particular shape/size.
- Category: recolor-conditional
- Shape: variable

### task365
- Rule: Grid contains two separate small multi-colored rectangular blob objects on an empty background; output is the exact bounding-box crop of one specific one of the two objects (selection appears based on object identity, not a fixed position).
- Category: object-detect
- Shape: variable

### task366
- Rule: Grid contains a large solid-color rectangle with a smaller distinct pattern embedded inside or beside it; output crops to that embedded pattern's bounding box, replacing the surrounding fill color with background 0.
- Category: object-detect
- Shape: variable

### task367
- Rule: Grid contains thin maze-like corridors of background cells bounded by walls of one color, with a couple of embedded dots; output floods a new color into the connected corridor path near those dots, leaving walls unchanged.
- Category: object-detect
- Shape: variable

### task368
- Rule: Grid has one multi-colored pattern object and one plain solid-color rectangle elsewhere; output recolors the plain rectangle's interior to replicate the multi-color pattern from the other object.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task369
- Rule: Grid is filled almost entirely with one color plus scattered background-0 cells; each 0 cell is recolored to 1, 2, or 3 depending on how many like-colored neighbors surround it (Minesweeper-style neighbor count).
- Category: counting
- Shape: constant 10x10 -> 10x10

### task370
- Rule: A tiny seed shape (2-3 marked cells) sits inside a solid-color field; output extends a diagonal ray outward from the seed, repeating the seed's local pattern cell-by-cell along the diagonal to the grid edge.
- Category: other-complex
- Shape: variable

### task371
- Rule: Two isolated single-color dots share a row (or are positioned symmetrically); output draws a small plus/cross shape of a new color (3) centered at the midpoint between them.
- Category: object-detect
- Shape: variable

### task372
- Rule: Grid is split by a full horizontal marker row (all one color) into an upper block and equal-size lower block holding complementary sparse patterns; output drops the marker row and overlays the upper and lower blocks row-for-row, keeping whichever cell is non-background. Verified exactly (266/266 matches) with an automated check.
- Category: symmetry-fill
- Shape: constant 11x11 -> 5x11

### task373
- Rule: A 2-row grid where each full row is one solid color; output turns both rows into an alternating checkerboard pattern using the same two colors, with the two rows out of phase with each other.
- Category: other-complex
- Shape: constant 2x6 -> 2x6

### task374
- Rule: Grid contains an L/step-shaped path made of one color; output recolors each straight segment of the path with a different new color depending on its position/order along the path.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task375
- Rule: Grid is one solid color with a single background-0 hole somewhere off-center; output creates concentric diamond (Manhattan-distance) rings of alternating 0/color radiating outward from that hole, tiling the whole grid.
- Category: other-complex
- Shape: variable

### task376
- Rule: Input is a short vertically-periodic (period-2 row) striped pattern; output continues that same alternating row pattern downward to a taller grid (height = 4*input_height - 3), same width.
- Category: tile-repeat
- Shape: variable

### task377
- Rule: A large mostly-empty grid contains one small nested concentric-square "target" pattern; output crops exactly to that pattern's bounding box.
- Category: object-detect
- Shape: variable

### task378
- Rule: Grid contains one small multi-color object near a corner; output adds a few new same-color pixels forming a short diagonal trail extending from the object toward the opposite corner.
- Category: object-detect
- Shape: variable

### task379
- Rule: Grid contains isolated marker dots plus one or two full-width/height "wall" lines of another color; output draws bracket/frame outlines around each dot, sized and clipped relative to the nearest wall line.
- Category: object-detect
- Shape: variable

### task380
- Rule: Rotates the grid 90 degrees counter-clockwise. Verified programmatically against all 267 available train/test/arc-gen examples with zero mismatches (unlike other candidate symmetric transforms which failed on some examples).
- Category: rotate90
- Shape: constant 3x3 -> 3x3

### task381
- Rule: Grid has blocky rectangular regions of one color with internal empty gaps that form enclosed pockets; output flood-fills those fully enclosed background pockets with a new color (9), leaving open (non-enclosed) background untouched.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task382
- Rule: A short dashed/dotted pattern occupies the first row along with a few scattered marker dots; output repeats/scrolls that dashed pattern down the full height, shifting it diagonally each block of rows.
- Category: other-complex
- Shape: variable

### task383
- Rule: Grid contains a large rectangular frame object with a small color anomaly on its border; output adds tick-mark/bracket lines extending outward from the frame at the position of the anomaly, projected through to the grid edges.
- Category: object-detect
- Shape: variable

### task384
- Rule: Grid contains one small blob object of a single color on an empty background; output crops to that object's bounding box and doubles its size (each cell becomes a 2x2 block).
- Category: object-detect
- Shape: variable

### task385
- Rule: A small colored shape occupies the bottom rows of the grid while the top rows are empty; output mirrors that shape upward so the whole grid becomes vertically symmetric (palindrome) about the boundary between the shape and the empty area.
- Category: symmetry-fill
- Shape: constant 10x4 -> 10x4

### task386
- Rule: Two small dot-patterns (colors 1 and 7) sit on either side of a vertical marker column (color 5); output is a narrower grid comparing the two patterns cell by cell, marking differences/matches with color 3.
- Category: other-complex
- Shape: constant 4x7 -> 4x3

### task387
- Rule: Grid contains pairs of small colored dots; output draws a nested box/frame around each dot and a connecting dashed line of a new color (5) linking the two frames.
- Category: object-detect
- Shape: variable

### task388
- Rule: Grid is tiled 2x2 into a grid twice as tall and wide; in the rows/positions that were entirely background in the original tile, output additionally paints color 8 at the columns where the original pattern had a non-background cell, creating a checkered highlight of the repeated motif.
- Category: other-complex
- Shape: variable

### task389
- Rule: Grid is filled with exactly two colors, one tracing a connected cross/diagonal "signal" shape and the other filling the remaining background cells; output keeps the signal-shape cells (recolored) and sets the rest to 0 (exact color-role selection rule not fully pinned down).
- Category: recolor-conditional
- Shape: variable

### task390
- Rule: Grid contains two separated copies of a small pattern sharing a color, one more complete than the other; output repairs/syncs one copy's cells using the mirrored version of the other so both match.
- Category: symmetry-fill
- Shape: constant 15x15 -> 15x15

### task391
- Rule: Grid contains several small multi-cell objects of different colors laid out on separate rows; output is a 3x1 column listing those colors, ordered by the size (cell count) of each object.
- Category: counting
- Shape: variable

### task392
- Rule: Grid contains a staircase/step boundary made of one color; output extends alternating stripes of that color and a new color (5) to fill the triangular region beyond the staircase, following the step pattern.
- Category: other-complex
- Shape: constant 10x10 -> 10x10

### task393
- Rule: Grid contains several small multi-cell blob objects of different colors; output is a 3x1 column listing those colors ordered by object size (cell count).
- Category: counting
- Shape: constant 12x12 -> 3x1

### task394
- Rule: Grid contains scattered cells of 2 (occasionally 3) colors; output is a small grid whose size scales with the counts of each color, effectively a compressed count/ratio summary of the color populations.
- Category: counting
- Shape: variable

### task395
- Rule: Grid is two stacked 3-row sub-grids: an upper one using colors {0,9} and a lower mask using colors {0,1}; output marks color 2 exactly where both sub-grids have a background-0 cell at that position, else 0. Verified against 2 train examples with an exact logical-AND-of-zeros rule.
- Category: symmetry-fill
- Shape: constant 6x3 -> 3x3

### task396
- Rule: A large grid contains one rectangular frame/box object drawn in a border color amid other noisy line clutter; output crops to that frame's bounding box, keeping only the frame's own color and background, discarding all other clutter.
- Category: object-detect
- Shape: variable

### task397
- Rule: Grid contains several small 2-cell colored objects; output draws a short diagonal connecting line of a new color (3) between certain pairs of nearby objects.
- Category: object-detect
- Shape: constant 10x10 -> 10x10

### task398
- Rule: Input is always a single row of 5 cells containing a short colored sequence; output tiles that same sequence repeated along successive diagonals to fill a larger square grid (size varies per example) anchored at the bottom-right corner.
- Category: other-complex
- Shape: variable

### task399
- Rule: Grid contains one or more separate blob objects of a single color; output is a fixed 3x3 grid with cells set to 1 (one per counted object, in a fixed scan order) representing the number of distinct objects, rest 0.
- Category: counting
- Shape: variable

### task400
- Rule: Large 24x24 grid with sparse noise on a mostly-repeating/symmetric pattern; output is the small 5x5 "clean" motif recovered from the pattern, similar in style to task351.
- Category: symmetry-fill
- Shape: constant 24x24 -> 5x5

