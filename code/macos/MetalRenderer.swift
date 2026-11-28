//
// Metal port: the rendering backend.
//
// mtl_qgl.c tracks the renderer's OpenGL state and hands each draw over as an
// mtl_state_t plus a packed vertex/index stream. This class turns those into
// Metal work: pipeline and depth-stencil states are created on demand and
// cached by key, vertex data goes into per-frame arenas, and the frame is
// drawn into an offscreen target at the engine's resolution, optionally
// upscaled with MetalFX (r_mtlUpscale), then scaled into the window and
// gamma-corrected at present time.
//

import AppKit
import Metal
import MetalFX
import QuartzCore

// MARK: - Per-frame buffer arena

/// Linear allocator over shared MTLBuffers. Reset once the GPU is done with the frame.
nonisolated private final class FrameArena {
    private let device: MTLDevice
    private var buffers: [MTLBuffer] = []
    private var current = 0
    private var offset = 0
    private static let chunkSize = 8 << 20

    init(device: MTLDevice) {
        self.device = device
    }

    func reset() {
        current = 0
        offset = 0
    }

    func allocate(_ length: Int) -> (MTLBuffer, Int)? {
        let length = (length + 255) & ~255
        while current < buffers.count {
            let buffer = buffers[current]
            if offset + length <= buffer.length {
                let result = (buffer, offset)
                offset += length
                return result
            }
            current += 1
            offset = 0
        }
        guard let buffer = device.makeBuffer(length: max(length, Self.chunkSize), options: .storageModeShared) else {
            return nil
        }
        buffer.label = "Q3 frame arena"
        buffers.append(buffer)
        current = buffers.count - 1
        offset = length
        return (buffer, 0)
    }
}

// MARK: - Frame completion

/// The newest frame the GPU has finished, written from Metal's completion threads.
nonisolated private final class CompletedFrame: @unchecked Sendable {
    private let lock = NSLock()
    private var frame = -1

    func mark(_ finished: Int) {
        lock.lock()
        frame = max(frame, finished)
        lock.unlock()
    }

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return frame
    }
}

// MARK: - Textures

nonisolated private struct TextureEntry {
    var texture: MTLTexture?
    var samplerKey: UInt16 = 0
    var lastUsedFrame: Int = -1
}

// MARK: - Renderer

// Not tied to an actor: the engine calls it on the main thread, and Metal
// runs its completion handlers on its own threads. (The macOS app target
// defaults to MainActor isolation.)
nonisolated final class MetalRenderer {
    nonisolated(unsafe) static private(set) var shared: MetalRenderer?

    /// r_mtlUpscale values.
    enum Upscale: Int {
        case off = 0, spatial, temporal, temporalDenoised
    }

    let device: MTLDevice
    let width: Int          // engine (internal) resolution
    let height: Int
    let outputWidth: Int    // after upscaling
    let outputHeight: Int
    private(set) var upscale: Upscale
    private let layer: CAMetalLayer
    private let queue: MTLCommandQueue
    private let library: MTLLibrary

    private let colorTarget: MTLTexture
    private let depthTarget: MTLTexture
    private var spatialScaler: MTLFXSpatialScaler?
    private var upscaledTarget: MTLTexture?
    private static let colorFormat = MTLPixelFormat.rgba8Unorm
    private static let depthFormat = MTLPixelFormat.depth32Float_stencil8

    private var pipelines: [UInt32: MTLRenderPipelineState] = [:]
    private var depthStates: [UInt64: MTLDepthStencilState] = [:]
    private var samplers: [UInt16: MTLSamplerState] = [:]
    private var clearPipelines: [Bool: MTLRenderPipelineState] = [:]
    private let presentPipeline: MTLRenderPipelineState

    private var textures: [TextureEntry] = []
    private let whiteTexture: MTLTexture
    private var gammaTexture: MTLTexture

    // frame pacing
    private static let framesInFlight = 3
    private let inFlight = DispatchSemaphore(value: MetalRenderer.framesInFlight)
    private var arenas: [FrameArena] = []
    private var frameNumber = 0
    private let completed = CompletedFrame()

    // current frame
    private var frameStarted = false
    private var clearOnLoad = false
    private var commandBuffer: MTLCommandBuffer?
    private var encoder: MTLRenderCommandEncoder?
    private var vertexAlloc: (MTLBuffer, Int)?
    private var indexAlloc: (MTLBuffer, Int)?

    // encoder state cache
    private var curPipeline: MTLRenderPipelineState?
    private var curDepthState: MTLDepthStencilState?
    private var curStencilRef: UInt32 = .max
    private var curCull: MTLCullMode?
    private var curFill: MTLTriangleFillMode?
    private var curBias: (Float, Float)?
    private var curViewport: MTLViewport?
    private var curScissor: MTLScissorRect?
    private var curTextures: [ObjectIdentifier?] = [nil, nil]
    private var curSamplers: [UInt16] = [.max, .max]
    private var curVertexBuffer: MTLBuffer?

    var swapInterval = 1 {
        didSet { layer.displaySyncEnabled = swapInterval != 0 }
    }

    // MARK: Setup

    init(layer: CAMetalLayer, width: Int, height: Int, outputWidth: Int, outputHeight: Int,
         upscale: Upscale) throws {
        guard let device = layer.device ?? MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw RendererError.noDevice
        }
        self.device = device
        self.queue = queue
        self.layer = layer
        self.width = width
        self.height = height
        self.outputWidth = upscale == .off ? width : outputWidth
        self.outputHeight = upscale == .off ? height : outputHeight
        self.upscale = upscale

        layer.device = device
        layer.pixelFormat = .bgra8Unorm
        layer.framebufferOnly = true
        layer.maximumDrawableCount = 3
        layer.allowsNextDrawableTimeout = true

        library = try Self.loadShaderLibrary(device: device)

        if upscale == .temporal || upscale == .temporalDenoised {
            // needs jitter and motion vectors from the front end (phase 3)
            MacPrint("MetalFX temporal upscaling is not implemented yet, using spatial\n")
            self.upscale = .spatial
        }
        if self.upscale == .spatial {
            spatialScaler = Self.makeSpatialScaler(device: device, inputWidth: width, inputHeight: height,
                                                   outputWidth: outputWidth, outputHeight: outputHeight)
            if spatialScaler == nil {
                MacPrint("MetalFX spatial upscaling is not available on this device\n")
                self.upscale = .off
            }
        }

        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: Self.colorFormat, width: width,
                                                                 height: height, mipmapped: false)
        colorDesc.usage = [.renderTarget, .shaderRead]
        if let spatialScaler {
            colorDesc.usage.formUnion(spatialScaler.colorTextureUsage)
        }
        colorDesc.storageMode = .private
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: Self.depthFormat, width: width,
                                                                 height: height, mipmapped: false)
        depthDesc.usage = .renderTarget
        depthDesc.storageMode = .private
        guard let color = device.makeTexture(descriptor: colorDesc),
              let depth = device.makeTexture(descriptor: depthDesc) else {
            throw RendererError.allocation
        }
        color.label = "Q3 frame color"
        depth.label = "Q3 frame depth/stencil"
        colorTarget = color
        depthTarget = depth

        if let spatialScaler {
            let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: Self.colorFormat,
                                                                   width: outputWidth, height: outputHeight,
                                                                   mipmapped: false)
            outDesc.usage = spatialScaler.outputTextureUsage.union(.shaderRead)
            outDesc.storageMode = .private
            guard let out = device.makeTexture(descriptor: outDesc) else {
                throw RendererError.allocation
            }
            out.label = "Q3 upscaled color"
            upscaledTarget = out
            MacPrint("MetalFX spatial upscaling \(width)x\(height) -> \(outputWidth)x\(outputHeight)\n")
        }

        let presentDesc = MTLRenderPipelineDescriptor()
        presentDesc.label = "present"
        presentDesc.vertexFunction = library.makeFunction(name: "present_vs")
        presentDesc.fragmentFunction = library.makeFunction(name: "present_fs")
        presentDesc.colorAttachments[0].pixelFormat = layer.pixelFormat
        presentPipeline = try device.makeRenderPipelineState(descriptor: presentDesc)

        whiteTexture = try Self.makeTexture(device: device, width: 1, height: 1, bytes: [255, 255, 255, 255])
        let ramp = (0..<256).flatMap { i -> [UInt8] in [UInt8(i), UInt8(i), UInt8(i), 255] }
        gammaTexture = try Self.makeTexture(device: device, width: 256, height: 1, bytes: ramp)

        arenas = (0..<Self.framesInFlight).map { _ in FrameArena(device: device) }
        MetalRenderer.shared = self
    }

    enum RendererError: Error {
        case noDevice
        case allocation
        case noShaderLibrary
    }

    /// Shaders.metal, compiled at build time: the app bundle's default.metallib,
    /// or for the stand-alone binary, default.metallib next to the executable.
    private static func loadShaderLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let library = try? device.makeDefaultLibrary(bundle: .main) {
            return library
        }
        if let exe = Bundle.main.executableURL?.resolvingSymlinksInPath() {
            let url = exe.deletingLastPathComponent().appendingPathComponent("default.metallib")
            if let library = try? device.makeLibrary(URL: url) {
                return library
            }
        }
        throw RendererError.noShaderLibrary
    }

    func shutdown() {
        // let the GPU finish before the textures and buffers go away
        if let buffer = commandBuffer {
            encoder?.endEncoding()
            buffer.commit()
            buffer.waitUntilCompleted()
        }
        if frameStarted {
            // this frame took a slot but will never present; give it back
            inFlight.signal()
            frameStarted = false
        }
        for _ in 0..<Self.framesInFlight { inFlight.wait() }
        for _ in 0..<Self.framesInFlight { inFlight.signal() }
        encoder = nil
        commandBuffer = nil
        textures.removeAll()
        MetalRenderer.shared = nil
    }

    private static func makeSpatialScaler(device: MTLDevice, inputWidth: Int, inputHeight: Int,
                                          outputWidth: Int, outputHeight: Int) -> MTLFXSpatialScaler? {
        guard MTLFXSpatialScalerDescriptor.supportsDevice(device) else { return nil }
        let desc = MTLFXSpatialScalerDescriptor()
        desc.inputWidth = inputWidth
        desc.inputHeight = inputHeight
        desc.outputWidth = outputWidth
        desc.outputHeight = outputHeight
        desc.colorTextureFormat = colorFormat
        desc.outputTextureFormat = colorFormat
        // the frame is gamma-encoded 8-bit, as the game drew it
        desc.colorProcessingMode = .perceptual
        return desc.makeSpatialScaler(device: device)
    }

    private static func makeTexture(device: MTLDevice, width: Int, height: Int, bytes: [UInt8]) throws -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height,
                                                            mipmapped: false)
        desc.usage = .shaderRead
        guard let texture = device.makeTexture(descriptor: desc) else {
            throw RendererError.allocation
        }
        texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: bytes,
                        bytesPerRow: width * 4)
        return texture
    }

    // MARK: Frame structure

    private func beginFrameIfNeeded() {
        if frameStarted { return }
        inFlight.wait()
        arenas[frameNumber % Self.framesInFlight].reset()
        frameStarted = true
        clearOnLoad = true
    }

    private func ensureCommandBuffer() -> MTLCommandBuffer? {
        beginFrameIfNeeded()
        if commandBuffer == nil {
            commandBuffer = queue.makeCommandBuffer()
            commandBuffer?.label = "Q3 frame \(frameNumber)"
        }
        return commandBuffer
    }

    private func ensureEncoder() -> MTLRenderCommandEncoder? {
        if let encoder { return encoder }
        guard let buffer = ensureCommandBuffer() else { return nil }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = colorTarget
        pass.colorAttachments[0].loadAction = clearOnLoad ? .clear : .load
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depthTarget
        pass.depthAttachment.loadAction = clearOnLoad ? .clear : .load
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.storeAction = .store
        pass.stencilAttachment.texture = depthTarget
        pass.stencilAttachment.loadAction = clearOnLoad ? .clear : .load
        pass.stencilAttachment.clearStencil = 0
        pass.stencilAttachment.storeAction = .store
        clearOnLoad = false

        guard let encoder = buffer.makeRenderCommandEncoder(descriptor: pass) else { return nil }
        encoder.label = "Q3 scene"
        encoder.setFrontFacing(.counterClockwise)
        self.encoder = encoder

        curPipeline = nil
        curDepthState = nil
        curStencilRef = .max
        curCull = nil
        curFill = nil
        curBias = nil
        curViewport = nil
        curScissor = nil
        curTextures = [nil, nil]
        curSamplers = [.max, .max]
        curVertexBuffer = nil
        return encoder
    }

    private func endEncoding() {
        encoder?.endEncoding()
        encoder = nil
    }

    func endFrame() {
        guard let buffer = ensureCommandBuffer() else { return }
        endEncoding()

        var presentSource = colorTarget
        if let spatialScaler, let upscaledTarget {
            spatialScaler.colorTexture = colorTarget
            spatialScaler.outputTexture = upscaledTarget
            spatialScaler.inputContentWidth = width
            spatialScaler.inputContentHeight = height
            spatialScaler.encode(commandBuffer: buffer)
            presentSource = upscaledTarget
        }

        let drawableSize = layer.drawableSize
        if drawableSize.width > 0, drawableSize.height > 0, let drawable = layer.nextDrawable() {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawable.texture
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            pass.colorAttachments[0].storeAction = .store
            if let enc = buffer.makeRenderCommandEncoder(descriptor: pass) {
                enc.label = "present"
                // fit the frame into the window, keeping its aspect ratio
                let dw = Double(drawableSize.width), dh = Double(drawableSize.height)
                let scale = min(dw / Double(outputWidth), dh / Double(outputHeight))
                let w = (Double(outputWidth) * scale).rounded(), h = (Double(outputHeight) * scale).rounded()
                enc.setViewport(MTLViewport(originX: ((dw - w) / 2).rounded(), originY: ((dh - h) / 2).rounded(),
                                            width: w, height: h, znear: 0, zfar: 1))
                enc.setRenderPipelineState(presentPipeline)
                enc.setFragmentTexture(presentSource, index: 0)
                enc.setFragmentTexture(gammaTexture, index: 1)
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                enc.endEncoding()
            }
            buffer.present(drawable)
        }

        let finished = frameNumber
        buffer.addCompletedHandler { [completed, inFlight] _ in
            completed.mark(finished)
            inFlight.signal()
        }
        buffer.commit()
        commandBuffer = nil
        frameStarted = false
        frameNumber += 1
    }

    /// Waits for everything submitted so far, mid-frame. Drawing continues afterwards
    /// in a new command buffer that loads the current contents.
    private func flushAndWait(_ encode: ((MTLCommandBuffer) -> Void)? = nil) {
        guard let buffer = ensureCommandBuffer() else { return }
        endEncoding()
        encode?(buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        commandBuffer = nil
    }

    func finish() {
        // Nothing needs the GPU to be idle here; work is ordered by the queue.
    }

    // MARK: Buffers

    func allocVertices(_ count: Int) -> UnsafeMutablePointer<mtl_vertex_t>? {
        beginFrameIfNeeded()
        guard count > 0,
              let alloc = arenas[frameNumber % Self.framesInFlight].allocate(count * MemoryLayout<mtl_vertex_t>.stride)
        else { return nil }
        vertexAlloc = alloc
        return (alloc.0.contents() + alloc.1).bindMemory(to: mtl_vertex_t.self, capacity: count)
    }

    func allocIndices(_ count: Int) -> UnsafeMutablePointer<UInt32>? {
        beginFrameIfNeeded()
        guard count > 0,
              let alloc = arenas[frameNumber % Self.framesInFlight].allocate(count * 4)
        else { return nil }
        indexAlloc = alloc
        return (alloc.0.contents() + alloc.1).bindMemory(to: UInt32.self, capacity: count)
    }

    // MARK: Drawing

    func draw(_ statePtr: UnsafePointer<mtl_state_t>, primitive: Int32, numVertexes: Int, numIndexes: Int) {
        guard let (vbuf, voff) = vertexAlloc, let enc = ensureEncoder() else { return }
        let s = statePtr.pointee
        vertexAlloc = nil

        applyRasterState(s, encoder: enc)

        // pipeline: blending, colour writes, alpha test
        let alphaTest = s.fs.alphaFunc != Int32(MTL_CMP_ALWAYS)
        let pipeKey = UInt32(s.blendEnable != 0 ? 1 : 0) | UInt32(s.blendSrc) << 1 | UInt32(s.blendDst) << 5
            | UInt32(s.colorMask & 15) << 9 | UInt32(alphaTest ? 1 : 0) << 13
        if let pipeline = pipeline(for: pipeKey, s: s, alphaTest: alphaTest), pipeline !== curPipeline {
            enc.setRenderPipelineState(pipeline)
            curPipeline = pipeline
        }

        applyDepthStencil(s, encoder: enc)

        // textures
        bindTexture(unit: 0, texnum: Int(s.tex0), enabled: s.fs.tex0Enable != 0, encoder: enc)
        bindTexture(unit: 1, texnum: Int(s.tex1), enabled: s.fs.tex1Enable != 0, encoder: enc)

        // vertices and uniforms
        if vbuf === curVertexBuffer {
            enc.setVertexBufferOffset(voff, index: 0)
        } else {
            enc.setVertexBuffer(vbuf, offset: voff, index: 0)
            curVertexBuffer = vbuf
        }
        let raw = UnsafeRawPointer(statePtr)
        enc.setVertexBytes(raw + MemoryLayout<mtl_state_t>.offset(of: \.vs)!,
                           length: MemoryLayout<mtl_vs_uniforms_t>.size, index: 1)
        enc.setFragmentBytes(raw + MemoryLayout<mtl_state_t>.offset(of: \.fs)!,
                             length: MemoryLayout<mtl_fs_uniforms_t>.size, index: 0)

        let type: MTLPrimitiveType
        switch Int(primitive) {
        case Int(MTL_PRIM_LINES): type = .line
        case Int(MTL_PRIM_POINTS): type = .point
        default: type = .triangle
        }

        if numIndexes > 0, let (ibuf, ioff) = indexAlloc {
            enc.drawIndexedPrimitives(type: type, indexCount: numIndexes, indexType: .uint32,
                                      indexBuffer: ibuf, indexBufferOffset: ioff)
        } else {
            enc.drawPrimitives(type: type, vertexStart: 0, vertexCount: numVertexes)
        }
        indexAlloc = nil
    }

    private func applyRasterState(_ s: mtl_state_t, encoder enc: MTLRenderCommandEncoder) {
        let cull: MTLCullMode
        switch Int(s.cullMode) {
        case Int(MTL_CULL_FRONT): cull = .front
        case Int(MTL_CULL_BACK): cull = .back
        default: cull = .none
        }
        if cull != curCull {
            enc.setCullMode(cull)
            curCull = cull
        }

        let fill: MTLTriangleFillMode = s.wireframe != 0 ? .lines : .fill
        if fill != curFill {
            enc.setTriangleFillMode(fill)
            curFill = fill
        }

        let bias: (Float, Float) = s.polygonOffset != 0 ? (s.offsetUnits, s.offsetFactor) : (0, 0)
        if curBias == nil || curBias! != bias {
            enc.setDepthBias(bias.0, slopeScale: bias.1, clamp: 0)
            curBias = bias
        }

        // OpenGL window coordinates have their origin at the bottom left
        let vp = s.viewport
        let viewport = MTLViewport(originX: Double(vp.0), originY: Double(height - Int(vp.1) - Int(vp.3)),
                                   width: Double(vp.2), height: Double(vp.3),
                                   znear: Double(s.depthNear), zfar: Double(s.depthFar))
        if curViewport.map({ !Self.same($0, viewport) }) ?? true {
            enc.setViewport(viewport)
            curViewport = viewport
        }

        let scissor = s.scissorEnable != 0
            ? clampedRect(x: Int(s.scissor.0), y: Int(s.scissor.1), w: Int(s.scissor.2), h: Int(s.scissor.3))
            : MTLScissorRect(x: 0, y: 0, width: width, height: height)
        if curScissor.map({ $0.x != scissor.x || $0.y != scissor.y || $0.width != scissor.width
                             || $0.height != scissor.height }) ?? true {
            enc.setScissorRect(scissor)
            curScissor = scissor
        }
    }

    private static func same(_ a: MTLViewport, _ b: MTLViewport) -> Bool {
        a.originX == b.originX && a.originY == b.originY && a.width == b.width && a.height == b.height
            && a.znear == b.znear && a.zfar == b.zfar
    }

    /// A GL-convention rectangle as a Metal scissor rect inside the frame.
    private func clampedRect(x: Int, y: Int, w: Int, h: Int) -> MTLScissorRect {
        let x0 = min(max(x, 0), width), x1 = min(max(x + w, 0), width)
        let top = height - (y + h), bottom = height - y
        let y0 = min(max(top, 0), height), y1 = min(max(bottom, 0), height)
        return MTLScissorRect(x: x0, y: y0, width: max(x1 - x0, 0), height: max(y1 - y0, 0))
    }

    private func pipeline(for key: UInt32, s: mtl_state_t, alphaTest: Bool) -> MTLRenderPipelineState? {
        if let p = pipelines[key] { return p }

        let constants = MTLFunctionConstantValues()
        var at = alphaTest
        constants.setConstantValue(&at, type: .bool, index: 0)

        let desc = MTLRenderPipelineDescriptor()
        desc.label = String(format: "q3 pipeline %04x", key)
        desc.vertexFunction = library.makeFunction(name: "q3_vs")
        desc.fragmentFunction = try? library.makeFunction(name: "q3_fs", constantValues: constants)
        desc.depthAttachmentPixelFormat = Self.depthFormat
        desc.stencilAttachmentPixelFormat = Self.depthFormat

        let ca = desc.colorAttachments[0]!
        ca.pixelFormat = Self.colorFormat
        var mask: MTLColorWriteMask = []
        if s.colorMask & 1 != 0 { mask.insert(.red) }
        if s.colorMask & 2 != 0 { mask.insert(.green) }
        if s.colorMask & 4 != 0 { mask.insert(.blue) }
        if s.colorMask & 8 != 0 { mask.insert(.alpha) }
        ca.writeMask = mask
        if s.blendEnable != 0 {
            ca.isBlendingEnabled = true
            ca.sourceRGBBlendFactor = Self.blendFactor(s.blendSrc)
            ca.sourceAlphaBlendFactor = Self.blendFactor(s.blendSrc)
            ca.destinationRGBBlendFactor = Self.blendFactor(s.blendDst)
            ca.destinationAlphaBlendFactor = Self.blendFactor(s.blendDst)
        }

        do {
            let p = try device.makeRenderPipelineState(descriptor: desc)
            pipelines[key] = p
            return p
        } catch {
            MacPrint("Metal: pipeline \(key) failed: \(error)\n")
            return nil
        }
    }

    private static func blendFactor(_ f: UInt8) -> MTLBlendFactor {
        switch Int32(f) {
        case Int32(MTL_BLEND_ZERO): return .zero
        case Int32(MTL_BLEND_ONE): return .one
        case Int32(MTL_BLEND_SRC_COLOR): return .sourceColor
        case Int32(MTL_BLEND_ONE_MINUS_SRC_COLOR): return .oneMinusSourceColor
        case Int32(MTL_BLEND_SRC_ALPHA): return .sourceAlpha
        case Int32(MTL_BLEND_ONE_MINUS_SRC_ALPHA): return .oneMinusSourceAlpha
        case Int32(MTL_BLEND_DST_COLOR): return .destinationColor
        case Int32(MTL_BLEND_ONE_MINUS_DST_COLOR): return .oneMinusDestinationColor
        case Int32(MTL_BLEND_DST_ALPHA): return .destinationAlpha
        case Int32(MTL_BLEND_ONE_MINUS_DST_ALPHA): return .oneMinusDestinationAlpha
        case Int32(MTL_BLEND_SRC_ALPHA_SATURATE): return .sourceAlphaSaturated
        default: return .one
        }
    }

    private static func stencilOp(_ op: UInt8) -> MTLStencilOperation {
        switch Int32(op) {
        case Int32(MTL_STENCIL_ZERO): return .zero
        case Int32(MTL_STENCIL_REPLACE): return .replace
        case Int32(MTL_STENCIL_INCR): return .incrementClamp
        case Int32(MTL_STENCIL_DECR): return .decrementClamp
        case Int32(MTL_STENCIL_INVERT): return .invert
        default: return .keep
        }
    }

    private func applyDepthStencil(_ s: mtl_state_t, encoder enc: MTLRenderCommandEncoder) {
        var key = UInt64(s.depthTest) | UInt64(s.depthWrite) << 1 | UInt64(s.depthFunc & 7) << 2
        if s.stencilTest != 0 {
            key |= 1 << 5 | UInt64(s.stencilFunc & 7) << 6 | UInt64(s.stencilFail & 7) << 9
                | UInt64(s.stencilZFail & 7) << 12 | UInt64(s.stencilZPass & 7) << 15
                | UInt64(s.stencilReadMask) << 18 | UInt64(s.stencilWriteMask) << 26
        }
        let state: MTLDepthStencilState
        if let cached = depthStates[key] {
            state = cached
        } else {
            let desc = MTLDepthStencilDescriptor()
            if s.depthTest != 0 {
                desc.depthCompareFunction = MTLCompareFunction(rawValue: UInt(s.depthFunc)) ?? .lessEqual
                desc.isDepthWriteEnabled = s.depthWrite != 0
            } else {
                desc.depthCompareFunction = .always
                desc.isDepthWriteEnabled = false
            }
            if s.stencilTest != 0 {
                let st = MTLStencilDescriptor()
                st.stencilCompareFunction = MTLCompareFunction(rawValue: UInt(s.stencilFunc)) ?? .always
                st.stencilFailureOperation = Self.stencilOp(s.stencilFail)
                st.depthFailureOperation = Self.stencilOp(s.stencilZFail)
                st.depthStencilPassOperation = Self.stencilOp(s.stencilZPass)
                st.readMask = UInt32(s.stencilReadMask)
                st.writeMask = UInt32(s.stencilWriteMask)
                desc.frontFaceStencil = st
                desc.backFaceStencil = st
            }
            guard let made = device.makeDepthStencilState(descriptor: desc) else { return }
            depthStates[key] = made
            state = made
        }
        if state !== curDepthState {
            enc.setDepthStencilState(state)
            curDepthState = state
        }
        if s.stencilTest != 0, UInt32(s.stencilRef) != curStencilRef {
            enc.setStencilReferenceValue(UInt32(s.stencilRef))
            curStencilRef = UInt32(s.stencilRef)
        }
    }

    private func bindTexture(unit: Int, texnum: Int, enabled: Bool, encoder enc: MTLRenderCommandEncoder) {
        var texture = whiteTexture
        var samplerKey: UInt16 = 0
        if enabled, texnum > 0, texnum < textures.count, let t = textures[texnum].texture {
            texture = t
            samplerKey = textures[texnum].samplerKey
            textures[texnum].lastUsedFrame = frameNumber
        }
        let id = ObjectIdentifier(texture)
        if curTextures[unit] != id {
            enc.setFragmentTexture(texture, index: unit)
            curTextures[unit] = id
        }
        if curSamplers[unit] != samplerKey {
            enc.setFragmentSamplerState(sampler(for: samplerKey), index: unit)
            curSamplers[unit] = samplerKey
        }
    }

    // MARK: Clears

    func clear(_ s: mtl_state_t, flags: Int32, color: (Float, Float, Float, Float), depth: Float, stencil: Int32) {
        guard let enc = ensureEncoder() else { return }
        let clearColor = flags & Int32(MTL_CLEAR_COLOR) != 0
        let clearDepth = flags & Int32(MTL_CLEAR_DEPTH) != 0
        let clearStencil = flags & Int32(MTL_CLEAR_STENCIL) != 0

        let pipeline: MTLRenderPipelineState
        if let p = clearPipelines[clearColor] {
            pipeline = p
        } else {
            let desc = MTLRenderPipelineDescriptor()
            desc.label = "clear"
            desc.vertexFunction = library.makeFunction(name: "clear_vs")
            desc.fragmentFunction = library.makeFunction(name: "clear_fs")
            desc.colorAttachments[0].pixelFormat = Self.colorFormat
            desc.colorAttachments[0].writeMask = clearColor ? .all : []
            desc.depthAttachmentPixelFormat = Self.depthFormat
            desc.stencilAttachmentPixelFormat = Self.depthFormat
            guard let p = try? device.makeRenderPipelineState(descriptor: desc) else { return }
            clearPipelines[clearColor] = p
            pipeline = p
        }
        enc.setRenderPipelineState(pipeline)
        curPipeline = pipeline

        let dsDesc = MTLDepthStencilDescriptor()
        dsDesc.depthCompareFunction = .always
        dsDesc.isDepthWriteEnabled = clearDepth
        if clearStencil {
            let st = MTLStencilDescriptor()
            st.stencilCompareFunction = .always
            st.depthStencilPassOperation = .replace
            st.writeMask = 0xff
            dsDesc.frontFaceStencil = st
            dsDesc.backFaceStencil = st
        }
        let dsKey = UInt64(1) << 40 | (clearDepth ? 1 : 0) | (clearStencil ? 2 : 0)
        let ds = depthStates[dsKey] ?? device.makeDepthStencilState(descriptor: dsDesc)!
        depthStates[dsKey] = ds
        enc.setDepthStencilState(ds)
        curDepthState = ds
        if clearStencil {
            enc.setStencilReferenceValue(UInt32(stencil & 0xff))
            curStencilRef = UInt32(stencil & 0xff)
        }

        if curCull != MTLCullMode.none {
            enc.setCullMode(.none)
            curCull = MTLCullMode.none
        }
        if curFill != .fill {
            enc.setTriangleFillMode(.fill)
            curFill = .fill
        }
        if curBias == nil || curBias! != (0, 0) {
            enc.setDepthBias(0, slopeScale: 0, clamp: 0)
            curBias = (0, 0)
        }
        // OpenGL clears ignore the viewport but honour the scissor
        let full = MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height), znear: 0, zfar: 1)
        enc.setViewport(full)
        curViewport = full
        let rect = s.scissorEnable != 0
            ? clampedRect(x: Int(s.scissor.0), y: Int(s.scissor.1), w: Int(s.scissor.2), h: Int(s.scissor.3))
            : MTLScissorRect(x: 0, y: 0, width: width, height: height)
        guard rect.width > 0, rect.height > 0 else { return }
        enc.setScissorRect(rect)
        curScissor = rect

        struct ClearUniforms {
            var color: SIMD4<Float>
            var depth: Float
        }
        var u = ClearUniforms(color: SIMD4(color.0, color.1, color.2, color.3), depth: min(max(depth, 0), 1))
        enc.setVertexBytes(&u, length: MemoryLayout<ClearUniforms>.stride, index: 0)
        enc.setFragmentBytes(&u, length: MemoryLayout<ClearUniforms>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        // slot 0 now holds the clear uniforms, not the vertex arena
        curVertexBuffer = nil
    }

    // MARK: Read back

    func readPixels(x: Int, y: Int, width w: Int, height h: Int, format: Int32, out: UnsafeMutableRawPointer) {
        let x0 = min(max(x, 0), width), y0 = min(max(y, 0), height)
        let w = min(w, width - x0), h = min(h, height - y0)
        guard w > 0, h > 0 else { return }

        let depthLike = format == Int32(MTL_READ_DEPTH) || format == Int32(MTL_READ_STENCIL)
        let texelSize = format == Int32(MTL_READ_STENCIL) ? 1 : 4
        let rowBytes = w * texelSize
        guard let staging = device.makeBuffer(length: rowBytes * h, options: .storageModeShared) else { return }

        let origin = MTLOrigin(x: x0, y: height - y0 - h, z: 0)
        let size = MTLSize(width: w, height: h, depth: 1)
        flushAndWait { buffer in
            guard let blit = buffer.makeBlitCommandEncoder() else { return }
            if depthLike {
                blit.copy(from: self.depthTarget, sourceSlice: 0, sourceLevel: 0, sourceOrigin: origin, sourceSize: size,
                          to: staging, destinationOffset: 0, destinationBytesPerRow: rowBytes,
                          destinationBytesPerImage: rowBytes * h,
                          options: format == Int32(MTL_READ_DEPTH) ? .depthFromDepthStencil
                                                                             : .stencilFromDepthStencil)
            } else {
                blit.copy(from: self.colorTarget, sourceSlice: 0, sourceLevel: 0, sourceOrigin: origin, sourceSize: size,
                          to: staging, destinationOffset: 0, destinationBytesPerRow: rowBytes,
                          destinationBytesPerImage: rowBytes * h)
            }
            blit.endEncoding()
        }

        // Metal rows run top to bottom; glReadPixels returns bottom to top.
        let src = staging.contents()
        for row in 0..<h {
            let from = src + (h - 1 - row) * rowBytes
            if format == Int32(MTL_READ_RGB) {
                let to = out + row * w * 3
                for i in 0..<w {
                    (to + i * 3).copyMemory(from: from + i * 4, byteCount: 3)
                }
            } else {
                (out + row * rowBytes).copyMemory(from: from, byteCount: rowBytes)
            }
        }
    }

    // MARK: Gamma

    func setGamma(red: UnsafePointer<UInt8>, green: UnsafePointer<UInt8>, blue: UnsafePointer<UInt8>) {
        var bytes = [UInt8](repeating: 255, count: 256 * 4)
        for i in 0..<256 {
            bytes[i * 4] = red[i]
            bytes[i * 4 + 1] = green[i]
            bytes[i * 4 + 2] = blue[i]
        }
        // a new texture, so frames still in flight keep the old ramp
        if let t = try? Self.makeTexture(device: device, width: 256, height: 1, bytes: bytes) {
            gammaTexture = t
        }
    }

    // MARK: Textures

    private func entryIndex(_ texnum: Int) -> Int? {
        guard texnum > 0, texnum < 1 << 16 else { return nil }
        if texnum >= textures.count {
            textures.append(contentsOf: repeatElement(TextureEntry(), count: texnum + 1 - textures.count))
        }
        return texnum
    }

    private var lastCompletedFrame: Int {
        completed.value
    }

    func texImage(_ texnum: Int, level: Int, width w: Int, height h: Int, data: UnsafeRawPointer, alphaOne: Bool) {
        guard let i = entryIndex(texnum), w > 0, h > 0 else { return }

        if level == 0 {
            // A new texture object every time level 0 is specified, so draws
            // still in flight keep reading the old contents.
            let levels = Int(log2(Double(max(w, h)))) + 1
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h,
                                                                mipmapped: levels > 1)
            desc.usage = .shaderRead
            desc.storageMode = .shared
            if alphaOne {
                desc.swizzle = MTLTextureSwizzleChannels(red: .red, green: .green, blue: .blue, alpha: .one)
            }
            guard let texture = device.makeTexture(descriptor: desc) else { return }
            texture.label = "q3 texture \(texnum)"
            textures[i].texture = texture
            textures[i].lastUsedFrame = -1
        }
        guard let texture = textures[i].texture, level < texture.mipmapLevelCount,
              w == max(texture.width >> level, 1), h == max(texture.height >> level, 1) else { return }
        texture.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: level, withBytes: data, bytesPerRow: w * 4)
    }

    func texSubImage(_ texnum: Int, level: Int, x: Int, y: Int, width w: Int, height h: Int, data: UnsafeRawPointer) {
        guard texnum > 0, texnum < textures.count, var texture = textures[texnum].texture,
              level < texture.mipmapLevelCount else { return }

        if textures[texnum].lastUsedFrame > lastCompletedFrame {
            // The GPU may still read this texture: update a copy instead (cinematics do this every frame).
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: texture.pixelFormat, width: texture.width,
                                                                height: texture.height,
                                                                mipmapped: texture.mipmapLevelCount > 1)
            desc.usage = .shaderRead
            desc.storageMode = .shared
            desc.swizzle = texture.swizzle
            guard let copy = device.makeTexture(descriptor: desc) else { return }
            copy.label = texture.label
            let covers = level == 0 && x == 0 && y == 0 && w == texture.width && h == texture.height
                && texture.mipmapLevelCount == 1
            if !covers {
                for l in 0..<texture.mipmapLevelCount {
                    let lw = max(texture.width >> l, 1), lh = max(texture.height >> l, 1)
                    var bytes = [UInt8](repeating: 0, count: lw * lh * 4)
                    texture.getBytes(&bytes, bytesPerRow: lw * 4, from: MTLRegionMake2D(0, 0, lw, lh), mipmapLevel: l)
                    copy.replace(region: MTLRegionMake2D(0, 0, lw, lh), mipmapLevel: l, withBytes: bytes,
                                 bytesPerRow: lw * 4)
                }
            }
            textures[texnum].texture = copy
            textures[texnum].lastUsedFrame = -1
            texture = copy
        }
        let lw = max(texture.width >> level, 1), lh = max(texture.height >> level, 1)
        guard x >= 0, y >= 0, x + w <= lw, y + h <= lh else { return }
        texture.replace(region: MTLRegionMake2D(x, y, w, h), mipmapLevel: level, withBytes: data, bytesPerRow: w * 4)
    }

    func texParams(_ texnum: Int, min: Int32, mag: Int32, mip: Int32, wrapS: Int32, wrapT: Int32) {
        guard let i = entryIndex(texnum) else { return }
        textures[i].samplerKey = UInt16(min & 1) | UInt16(mag & 1) << 1 | UInt16(mip & 3) << 2
            | UInt16(wrapS & 1) << 4 | UInt16(wrapT & 1) << 5 | 1 << 8
    }

    func deleteTexture(_ texnum: Int) {
        guard texnum > 0, texnum < textures.count else { return }
        textures[texnum] = TextureEntry()
    }

    private func sampler(for key: UInt16) -> MTLSamplerState {
        if let s = samplers[key] { return s }
        let desc = MTLSamplerDescriptor()
        if key == 0 {
            // never configured: OpenGL's defaults are close enough for the white texture
            desc.minFilter = .linear
            desc.magFilter = .linear
        } else {
            desc.minFilter = key & 1 != 0 ? .linear : .nearest
            desc.magFilter = key & 2 != 0 ? .linear : .nearest
            switch (key >> 2) & 3 {
            case UInt16(MTL_MIP_NEAREST): desc.mipFilter = .nearest
            case UInt16(MTL_MIP_LINEAR): desc.mipFilter = .linear
            default: desc.mipFilter = .notMipmapped
            }
            desc.sAddressMode = key & 16 != 0 ? .clampToEdge : .repeat
            desc.tAddressMode = key & 32 != 0 ? .clampToEdge : .repeat
        }
        let s = device.makeSamplerState(descriptor: desc)!
        samplers[key] = s
        return s
    }
}

// MARK: - Console output

nonisolated func MacPrint(_ message: String) {
    message.withCString { MacQ_Print($0) }
}

// MARK: - C entry points (declared in mtl_swift.h)

@_cdecl("MTLR_AllocVertices")
public func MTLR_AllocVertices(_ count: Int32) -> UnsafeMutablePointer<mtl_vertex_t>? {
    MetalRenderer.shared?.allocVertices(Int(count))
}

@_cdecl("MTLR_AllocIndices")
public func MTLR_AllocIndices(_ count: Int32) -> UnsafeMutablePointer<UInt32>? {
    MetalRenderer.shared?.allocIndices(Int(count))
}

@_cdecl("MTLR_Draw")
public func MTLR_Draw(_ state: UnsafePointer<mtl_state_t>, _ primitive: Int32, _ numVertexes: Int32, _ numIndexes: Int32) {
    MetalRenderer.shared?.draw(state, primitive: primitive, numVertexes: Int(numVertexes), numIndexes: Int(numIndexes))
}

@_cdecl("MTLR_Clear")
public func MTLR_Clear(_ state: UnsafePointer<mtl_state_t>, _ flags: Int32, _ color: UnsafePointer<Float>,
                       _ depth: Float, _ stencil: Int32) {
    MetalRenderer.shared?.clear(state.pointee, flags: flags, color: (color[0], color[1], color[2], color[3]),
                                depth: depth, stencil: stencil)
}

@_cdecl("MTLR_ReadPixels")
public func MTLR_ReadPixels(_ x: Int32, _ y: Int32, _ width: Int32, _ height: Int32, _ format: Int32,
                            _ out: UnsafeMutableRawPointer) {
    MetalRenderer.shared?.readPixels(x: Int(x), y: Int(y), width: Int(width), height: Int(height), format: format,
                                     out: out)
}

@_cdecl("MTLR_EndFrame")
public func MTLR_EndFrame() {
    MetalRenderer.shared?.endFrame()
}

@_cdecl("MTLR_Finish")
public func MTLR_Finish() {
    MetalRenderer.shared?.finish()
}

@_cdecl("MTLR_SetGamma")
public func MTLR_SetGamma(_ red: UnsafePointer<UInt8>, _ green: UnsafePointer<UInt8>, _ blue: UnsafePointer<UInt8>) {
    MetalRenderer.shared?.setGamma(red: red, green: green, blue: blue)
}

@_cdecl("MTLR_TexImage")
public func MTLR_TexImage(_ texnum: Int32, _ level: Int32, _ width: Int32, _ height: Int32, _ data: UnsafeRawPointer,
                          _ alphaOne: Bool) {
    MetalRenderer.shared?.texImage(Int(texnum), level: Int(level), width: Int(width), height: Int(height), data: data,
                                   alphaOne: alphaOne)
}

@_cdecl("MTLR_TexSubImage")
public func MTLR_TexSubImage(_ texnum: Int32, _ level: Int32, _ x: Int32, _ y: Int32, _ width: Int32, _ height: Int32,
                             _ data: UnsafeRawPointer) {
    MetalRenderer.shared?.texSubImage(Int(texnum), level: Int(level), x: Int(x), y: Int(y), width: Int(width),
                                      height: Int(height), data: data)
}

@_cdecl("MTLR_TexParams")
public func MTLR_TexParams(_ texnum: Int32, _ minFilter: Int32, _ magFilter: Int32, _ mipFilter: Int32,
                           _ wrapS: Int32, _ wrapT: Int32) {
    MetalRenderer.shared?.texParams(Int(texnum), min: minFilter, mag: magFilter, mip: mipFilter, wrapS: wrapS,
                                    wrapT: wrapT)
}

@_cdecl("MTLR_DeleteTexture")
public func MTLR_DeleteTexture(_ texnum: Int32) {
    MetalRenderer.shared?.deleteTexture(Int(texnum))
}
