//
// Metal port: the game window and AppKit event handling.
//
// The engine's main loop runs on the main thread and calls MacPlat_PumpEvents
// once per frame (Sys_SendKeyEvents). Events are pulled from the application
// queue here and reported to the engine through the MacQ_* callbacks in
// mac_glimp.c; the rest go to AppKit as usual. The same code serves the
// stand-alone quake3-metal binary and the macOS app, which runs the engine
// from inside its own run loop.
//

import AppKit
import CoreGraphics
import QuartzCore

/// The content view: hosts the CAMetalLayer and keeps its drawable size in pixels.
private final class GameView: NSView {
    let metalLayer = CAMetalLayer()

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer = metalLayer
        metalLayer.contentsGravity = .resizeAspect
        updateDrawableSize()
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    override var acceptsFirstResponder: Bool { true }
    override var isOpaque: Bool { true }

    // Key events are consumed in the pump; do not let AppKit beep about them.
    override func keyDown(with event: NSEvent) {}
    override func keyUp(with event: NSEvent) {}

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        updateDrawableSize()
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        updateDrawableSize()
    }

    func updateDrawableSize() {
        let scale = window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 1
        metalLayer.contentsScale = scale
        metalLayer.drawableSize = CGSize(width: bounds.width * scale, height: bounds.height * scale)
    }
}

/// A borderless window cannot become key by default, which would leave the
/// fullscreen window without keyboard focus or a mouse grab.
private final class GameWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

final class Platform: NSObject, NSWindowDelegate {
    static private(set) var shared: Platform?

    private let window: NSWindow
    private let view: GameView
    private let fullscreen: Bool
    let renderer: MetalRenderer

    private var wantGrab = false
    private var grabbed = false
    private var mouseRemainder = CGPoint.zero
    private var scrollRemainder: CGFloat = 0
    private var keysDown = Set<UInt16>()          // key codes the engine saw go down
    private var appObservers: [NSObjectProtocol] = []

    // MARK: Setup

    init(width: Int, height: Int, renderWidth: Int, renderHeight: Int, upscale: Int, fullscreen: Bool) throws {
        Platform.prepareApplication()
        self.fullscreen = fullscreen

        let screen = NSScreen.main ?? NSScreen.screens.first
        let contentRect: NSRect
        let style: NSWindow.StyleMask
        if fullscreen, let screen {
            contentRect = screen.frame
            style = [.borderless]
        } else {
            contentRect = NSRect(x: 0, y: 0, width: width, height: height)
            style = [.titled, .closable, .miniaturizable]
        }
        window = GameWindow(contentRect: contentRect, styleMask: style, backing: .buffered, defer: false, screen: screen)
        window.title = "Quake III Arena"
        window.isReleasedWhenClosed = false
        window.acceptsMouseMovedEvents = true
        window.backgroundColor = .black
        window.collectionBehavior = [.fullScreenNone]
        if fullscreen {
            window.setFrame(contentRect, display: false)
        } else {
            window.center()
        }

        view = GameView(frame: NSRect(origin: .zero, size: contentRect.size))
        window.contentView = view
        view.metalLayer.device = MTLCreateSystemDefaultDevice()
        renderer = try MetalRenderer(layer: view.metalLayer, width: renderWidth, height: renderHeight,
                                     outputWidth: width, outputHeight: height,
                                     upscale: MetalRenderer.Upscale(rawValue: upscale) ?? .off)

        super.init()
        window.delegate = self
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)
        view.updateDrawableSize()
        setFullscreenPresentation(true)

        // Switching apps (Cmd-Tab, clicking another app) must hand the mouse,
        // keyboard and screen back; coming back takes them again.
        let center = NotificationCenter.default
        appObservers.append(center.addObserver(forName: NSApplication.didResignActiveNotification, object: nil,
                                               queue: nil) { [weak self] _ in self?.appDidResignActive() })
        appObservers.append(center.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil,
                                               queue: nil) { [weak self] _ in self?.appDidBecomeActive() })
        NSApp.activate()
        Platform.shared = self
    }

    /// The stand-alone binary has no NSApplication yet; the macOS app has one running.
    private static func prepareApplication() {
        let app = NSApplication.shared
        if app.mainMenu == nil {
            app.setActivationPolicy(.regular)
            let mainMenu = NSMenu()
            let appItem = NSMenuItem()
            mainMenu.addItem(appItem)
            let appMenu = NSMenu()
            let quit = NSMenuItem(title: "Quit Quake III Arena", action: #selector(Platform.quitFromMenu(_:)),
                                  keyEquivalent: "q")
            quit.target = Platform.self
            appMenu.addItem(quit)
            appItem.submenu = appMenu
            app.mainMenu = mainMenu
        }
        if !app.isRunning {
            app.finishLaunching()
        }
    }

    @objc static func quitFromMenu(_ sender: Any?) {
        MacQ_Quit()
    }

    func shutdown() {
        for observer in appObservers {
            NotificationCenter.default.removeObserver(observer)
        }
        appObservers.removeAll()
        applyGrab(false)
        releaseAllKeys()
        renderer.shutdown()
        setFullscreenPresentation(false)
        window.delegate = nil
        window.orderOut(nil)
        window.close()
        Platform.shared = nil
    }

    // MARK: Focus

    /// In fullscreen, the window covers the menu bar and the Dock is hidden only
    /// while the game is the active app.
    private func setFullscreenPresentation(_ on: Bool) {
        guard fullscreen else { return }
        window.level = on ? .mainMenu + 1 : .normal
        NSApp.presentationOptions = on ? [.hideDock, .hideMenuBar] : []
    }

    private func appDidResignActive() {
        applyGrab(false)
        releaseAllKeys()
        setFullscreenPresentation(false)
        if fullscreen {
            // step out of the way of the app being switched to
            window.orderBack(nil)
        }
    }

    private func appDidBecomeActive() {
        setFullscreenPresentation(true)
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)
        applyGrab(wantGrab && window.isKeyWindow)
    }

    // MARK: Mouse grab

    func setGrab(_ on: Bool) {
        wantGrab = on
        applyGrab(on && window.isKeyWindow && NSApp.isActive)
    }

    private func applyGrab(_ on: Bool) {
        guard on != grabbed else { return }
        grabbed = on
        if on {
            NSCursor.hide()
            CGAssociateMouseAndMouseCursorPosition(0)
            // park the pointer in the middle of the window so clicks land there
            if let screenHeight = NSScreen.screens.first?.frame.height {
                let f = window.frame
                CGWarpMouseCursorPosition(CGPoint(x: f.midX, y: screenHeight - f.midY))
            }
            mouseRemainder = .zero
        } else {
            CGAssociateMouseAndMouseCursorPosition(1)
            NSCursor.unhide()
        }
    }

    func windowDidBecomeKey(_ notification: Notification) {
        applyGrab(wantGrab)
    }

    func windowDidResignKey(_ notification: Notification) {
        applyGrab(false)
        releaseAllKeys()
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        MacQ_Quit()
        return false
    }

    // MARK: Events

    func pumpEvents() {
        while let event = NSApp.nextEvent(matching: .any, until: .distantPast, inMode: .default, dequeue: true) {
            if !handle(event) {
                NSApp.sendEvent(event)
            }
        }
    }

    /// Returns true when the event was consumed by the game.
    private func handle(_ event: NSEvent) -> Bool {
        let inGameWindow = event.window === window
        switch event.type {
        case .keyDown:
            if event.modifierFlags.contains(.command) { return false }   // menu shortcuts
            guard window.isKeyWindow else { return false }
            let ch = event.characters(byApplyingModifiers: [])?.unicodeScalars.first.map { Int32($0.value) } ?? 0
            MacQ_KeyEvent(Int32(event.keyCode), ch, 1)
            keysDown.insert(event.keyCode)
            if let text = event.characters {
                for scalar in text.unicodeScalars where scalar.value < 0xF700 || scalar.value > 0xF8FF {
                    MacQ_CharEvent(Int32(scalar.value))
                }
            }
            return true

        case .keyUp:
            keysDown.remove(event.keyCode)
            MacQ_KeyEvent(Int32(event.keyCode), 0, 0)
            return !event.modifierFlags.contains(.command)

        case .flagsChanged:
            let flag: NSEvent.ModifierFlags
            switch Int(event.keyCode) {
            case 0x38, 0x3C: flag = .shift
            case 0x3B, 0x3E: flag = .control
            case 0x3A, 0x3D: flag = .option
            case 0x37, 0x36: flag = .command
            case 0x39: flag = .capsLock
            default: return false
            }
            let down = event.modifierFlags.contains(flag)
            if down { keysDown.insert(event.keyCode) } else { keysDown.remove(event.keyCode) }
            MacQ_KeyEvent(Int32(event.keyCode), 0, down ? 1 : 0)
            return false

        case .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            guard grabbed else { return false }
            let x = event.deltaX + mouseRemainder.x, y = event.deltaY + mouseRemainder.y
            let dx = x.rounded(.towardZero), dy = y.rounded(.towardZero)
            mouseRemainder = CGPoint(x: x - dx, y: y - dy)
            MacQ_MouseMove(Int32(dx), Int32(dy))
            return true

        case .leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp, .otherMouseDown, .otherMouseUp:
            guard grabbed || (inGameWindow && isInContent(event)) else { return false }
            let button: Int32
            switch event.type {
            case .leftMouseDown, .leftMouseUp: button = 0
            case .rightMouseDown, .rightMouseUp: button = 1
            default: button = event.buttonNumber == 2 ? 2 : Int32(min(event.buttonNumber, 4))
            }
            let down: Int32 = [.leftMouseDown, .rightMouseDown, .otherMouseDown].contains(event.type) ? 1 : 0
            MacQ_MouseButton(button, down)
            // let AppKit see clicks too while the pointer is free (focus, window activation)
            return grabbed

        case .scrollWheel:
            guard grabbed || inGameWindow else { return false }
            if event.hasPreciseScrollingDeltas {
                scrollRemainder += event.scrollingDeltaY
                while abs(scrollRemainder) >= 16 {
                    MacQ_MouseWheel(scrollRemainder > 0 ? 1 : -1)
                    scrollRemainder -= scrollRemainder > 0 ? 16 : -16
                }
            } else if event.scrollingDeltaY != 0 {
                MacQ_MouseWheel(event.scrollingDeltaY > 0 ? 1 : -1)
            }
            return true

        default:
            return false
        }
    }

    private func isInContent(_ event: NSEvent) -> Bool {
        view.bounds.contains(view.convert(event.locationInWindow, from: nil))
    }

    /// Focus moved away: the key releases will go to another app, so release
    /// everything the engine still thinks is held (movement keys included).
    private func releaseAllKeys() {
        for code in keysDown {
            MacQ_KeyEvent(Int32(code), 0, 0)
        }
        keysDown.removeAll()
    }
}

// MARK: - C entry points (declared in mtl_swift.h)

@_cdecl("MacPlat_InitVideo")
public func MacPlat_InitVideo(_ width: Int32, _ height: Int32, _ renderWidth: Int32, _ renderHeight: Int32,
                              _ upscale: Int32, _ fullscreen: Bool,
                              _ rendererName: UnsafeMutablePointer<CChar>?, _ nameSize: Int32) -> Bool {
    Platform.shared?.shutdown()
    do {
        let platform = try Platform(width: Int(width), height: Int(height), renderWidth: Int(renderWidth),
                                    renderHeight: Int(renderHeight), upscale: Int(upscale), fullscreen: fullscreen)
        if let rendererName, nameSize > 0 {
            let name = "Metal: \(platform.renderer.device.name)"
            _ = name.withCString { strlcpy(rendererName, $0, Int(nameSize)) }
        }
        return true
    } catch {
        MacPrint("Metal initialisation failed: \(error)\n")
        return false
    }
}

@_cdecl("MacPlat_ShutdownVideo")
public func MacPlat_ShutdownVideo() {
    Platform.shared?.shutdown()
}

@_cdecl("MacPlat_PumpEvents")
public func MacPlat_PumpEvents() {
    Platform.shared?.pumpEvents()
}

@_cdecl("MacPlat_GrabMouse")
public func MacPlat_GrabMouse(_ grab: Bool) {
    Platform.shared?.setGrab(grab)
}

@_cdecl("MacPlat_SetSwapInterval")
public func MacPlat_SetSwapInterval(_ interval: Int32) {
    Platform.shared?.renderer.swapInterval = Int(interval)
}
