// Selection-highlight test: voxels whose palette slot falls in [highlight.y,
// highlightMax] are "selected"; highlightEdge marks their XZ silhouette.

import package::uniforms::U;
import package::grid::sampleVoxel;

fn isHighlightCell(p : vec3<i32>) -> bool {
  let c = sampleVoxel(p);
  return c >= u32(U.highlight.y) && c <= u32(U.highlightMax);
}

// 1.0 if a highlighted cell sits on the selection's silhouette (an XZ neighbour
// is not highlighted) — flat top-layer objects read their edge in the XZ plane.
fn highlightEdge(cell : vec3<i32>) -> f32 {
  let l = isHighlightCell(cell + vec3<i32>(-1, 0, 0));
  let r = isHighlightCell(cell + vec3<i32>(1, 0, 0));
  let f = isHighlightCell(cell + vec3<i32>(0, 0, -1));
  let b = isHighlightCell(cell + vec3<i32>(0, 0, 1));
  return select(1.0, 0.0, l && r && f && b);
}
