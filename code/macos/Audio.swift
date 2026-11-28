//
// Metal port: sound output.
//
// The engine mixes into a ring buffer of interleaved 16-bit samples
// (mac_snd.c). An AVAudioSourceNode pulls from it on the audio thread,
// converts to the engine's float format and advances the shared read
// position, which the mixer uses to stay ahead of playback.
//

import AVFoundation

// The render block runs on the audio thread, so nothing here is MainActor
// isolated (the macOS app target defaults to MainActor).
nonisolated private final class AudioOutput {
    private let engine = AVAudioEngine()
    private let source: AVAudioSourceNode

    init(buffer: UnsafeMutablePointer<Int16>, samples: Int, channels: Int, rate: Double,
         position: UnsafeMutablePointer<Int32>) throws {
        guard let format = AVAudioFormat(standardFormatWithSampleRate: rate, channels: AVAudioChannelCount(channels))
        else {
            throw NSError(domain: "Quake3Audio", code: 1)
        }
        let scale: Float = 1.0 / 32768.0
        source = AVAudioSourceNode(format: format) { _, _, frameCount, audioBufferList -> OSStatus in
            let outputs = UnsafeMutableAudioBufferListPointer(audioBufferList)
            var pos = Int(position.pointee)
            if pos < 0 || pos >= samples { pos = 0 }
            for frame in 0..<Int(frameCount) {
                for channel in 0..<channels {
                    if channel < outputs.count, let data = outputs[channel].mData {
                        data.assumingMemoryBound(to: Float.self)[frame] = Float(buffer[pos]) * scale
                    }
                    pos += 1
                    if pos >= samples { pos = 0 }
                }
            }
            position.pointee = Int32(pos)
            return noErr
        }
        engine.attach(source)
        if #available(macOS 27, *) {
            try engine.connectNode(source, to: engine.mainMixerNode, format: format)
        } else {
            engine.connect(source, to: engine.mainMixerNode, format: format)
        }
        try engine.start()
    }

    func stop() {
        engine.stop()
    }
}

private var output: AudioOutput?

// MARK: - C entry points (declared in mtl_swift.h)

@_cdecl("MacAudio_Start")
public func MacAudio_Start(_ buffer: UnsafeMutablePointer<Int16>, _ samples: Int32, _ channels: Int32, _ rate: Int32,
                           _ position: UnsafeMutablePointer<Int32>) -> Bool {
    output?.stop()
    do {
        output = try AudioOutput(buffer: buffer, samples: Int(samples), channels: Int(channels), rate: Double(rate),
                                 position: position)
        return true
    } catch {
        MacPrint("Audio output failed: \(error)\n")
        output = nil
        return false
    }
}

@_cdecl("MacAudio_Stop")
public func MacAudio_Stop() {
    output?.stop()
    output = nil
}
