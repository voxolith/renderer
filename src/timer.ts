// GPU time per pass, from timestamp queries (the optional `timestamp-query` feature).
//
// Each named pass writes a begin and an end timestamp; after the frame's commands the queries are
// resolved and copied into one of a few readback buffers, mapped asynchronously and folded into a
// running average. A frame whose readback buffers are all still in flight is simply not measured,
// so timing never stalls rendering. Browsers quantise timestamps (to about 100 µs unless
// cross-origin isolated), which the averaging smooths out.

/** Measures named GPU passes; see {@link GpuTimer.passWrites} and {@link GpuTimer.timings}. */
export class GpuTimer {
  private readonly set: GPUQuerySet;
  private readonly resolveBuf: GPUBuffer;
  private readonly readBufs: { buf: GPUBuffer; busy: boolean; names: string[] }[] = [];
  private names: string[] = [];
  private readonly avg = new Map<string, number>();

  /**
   * @param device - A device created with the `timestamp-query` feature.
   * @param maxPasses - Passes measured per frame at most.
   */
  constructor(device: GPUDevice, private readonly maxPasses = 8) {
    this.set = device.createQuerySet({ type: "timestamp", count: maxPasses * 2 });
    this.resolveBuf = device.createBuffer({ size: maxPasses * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    for (let i = 0; i < 3; i++) {
      this.readBufs.push({ buf: device.createBuffer({ size: maxPasses * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false, names: [] });
    }
  }

  /** Start a frame: forget which passes were named last frame. */
  begin(): void {
    this.names = [];
  }

  /** The `timestampWrites` for a pass named `name` this frame, or undefined when out of slots. */
  passWrites(name: string): GPURenderPassTimestampWrites | GPUComputePassTimestampWrites | undefined {
    const i = this.names.length;
    if (i >= this.maxPasses) return undefined;
    this.names.push(name);
    return { querySet: this.set, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 };
  }

  /**
   * Resolve this frame's timestamps into a free readback buffer (call on the frame's encoder,
   * last); returns a function to call after `queue.submit` that starts the readback.
   */
  resolve(encoder: GPUCommandEncoder): () => void {
    const n = this.names.length;
    const slot = this.readBufs.find((r) => !r.busy);
    if (!n || !slot) return () => {};
    encoder.resolveQuerySet(this.set, 0, n * 2, this.resolveBuf, 0);
    encoder.copyBufferToBuffer(this.resolveBuf, 0, slot.buf, 0, n * 16);
    slot.busy = true;
    slot.names = this.names.slice();
    return () => {
      slot.buf.mapAsync(GPUMapMode.READ, 0, n * 16).then(
        () => {
          const t = new BigUint64Array(slot.buf.getMappedRange(0, n * 16).slice(0));
          slot.buf.unmap();
          slot.busy = false;
          for (let k = 0; k < slot.names.length; k++) {
            const ms = Number(t[k * 2 + 1] - t[k * 2]) / 1e6;
            if (!(ms >= 0 && ms < 1000)) continue;
            const name = slot.names[k];
            const prev = this.avg.get(name);
            this.avg.set(name, prev === undefined ? ms : prev * 0.9 + ms * 0.1);
          }
        },
        () => { slot.busy = false; },
      );
    };
  }

  /** Average GPU milliseconds per pass, by name (empty until the first readback lands). */
  timings(): Record<string, number> {
    return Object.fromEntries(this.avg);
  }
}
