//
//  AppDelegate.swift
//  quake3
//
//  Created by Jocelyn Manger on 11/24/26.
//

import Cocoa

@main
class AppDelegate: NSObject, NSApplicationDelegate {

    @IBOutlet var window: NSWindow!

    private static let bookmarkKey = "GameFolderBookmark"
    private var engineRunning = false

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        // Hold Option while launching to pick a different game folder.
        let forcePick = NSEvent.modifierFlags.contains(.option)
        guard let gameFolder = (forcePick ? nil : restoreGameFolder()) ?? chooseGameFolder() else {
            NSApp.terminate(nil)
            return
        }

        // Start the engine once launching has finished. Q3_Main never returns:
        // its loop runs on the main thread and pumps AppKit events itself
        // (code/macos/Platform.swift) and renders with Metal.
        installBundledPaks()
        DispatchQueue.main.async {
            self.runEngine(basePath: gameFolder.path)
        }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // Cmd-Q: let the engine shut down cleanly; it exits the process itself.
        if engineRunning {
            Q3_RequestQuit()
            return .terminateCancel
        }
        return .terminateNow
    }

    func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {
        return true
    }

    // MARK: - Engine

    private func runEngine(basePath: String) {
        // Pass through only Quake-style arguments ("+set ...", "+devmap ...");
        // Xcode adds its own (-NSDocumentRevisionsDebugMode etc.) before them.
        var args = CommandLine.arguments
        let userArgs = args.dropFirst().drop { !$0.hasPrefix("+") }
        args = [args[0], "+set", "fs_basepath", basePath] + userArgs

        var argv: [UnsafeMutablePointer<CChar>?] = args.map { strdup($0) }
        argv.append(nil)
        engineRunning = true
        _ = Q3_Main(Int32(args.count), &argv)
    }

    // MARK: - Bundled paks

    /// Copies the paks built with the app (zz_metal_ui.pk3: the menus with the
    /// Metal options) into the engine's home folder. The game folder is
    /// read-only to the sandboxed app, and paks in the home folder override
    /// the game folder's, so this is where they take effect.
    private func installBundledPaks() {
        // the engine's default fs_homepath (Sys_DefaultHomePath), inside the container
        let home = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Application Support/Quake3/baseq3", isDirectory: true)
        let fm = FileManager.default
        guard let paks = Bundle.main.urls(forResourcesWithExtension: "pk3", subdirectory: nil), !paks.isEmpty else {
            return
        }
        do {
            try fm.createDirectory(at: home, withIntermediateDirectories: true)
            for pak in paks {
                let dest = home.appendingPathComponent(pak.lastPathComponent)
                if fm.contentsEqual(atPath: pak.path, andPath: dest.path) { continue }
                try? fm.removeItem(at: dest)
                try fm.copyItem(at: pak, to: dest)
            }
        } catch {
            NSLog("quake3: could not install bundled paks: \(error)")
        }
    }

    // MARK: - Game folder

    /// A folder that holds baseq3/ with at least one .pk3.
    private static func isGameFolder(_ url: URL) -> Bool {
        let base = url.appendingPathComponent("baseq3")
        let files = (try? FileManager.default.contentsOfDirectory(atPath: base.path)) ?? []
        return files.contains { $0.lowercased().hasSuffix(".pk3") }
    }

    private func restoreGameFolder() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: Self.bookmarkKey) else { return nil }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: data,
                                 options: .withSecurityScope,
                                 relativeTo: nil,
                                 bookmarkDataIsStale: &stale),
              url.startAccessingSecurityScopedResource() else {
            return nil
        }
        guard Self.isGameFolder(url) else {
            url.stopAccessingSecurityScopedResource()
            return nil
        }
        if stale {
            saveBookmark(for: url)
        }
        return url
    }

    private func chooseGameFolder() -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Choose Quake III Arena Folder"
        panel.message = "Choose the folder that contains baseq3 (with pak0.pk3)."
        panel.prompt = "Choose"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false

        while panel.runModal() == .OK, var url = panel.url {
            // Accept the baseq3 folder itself as well as its parent.
            if url.lastPathComponent.lowercased() == "baseq3" && !Self.isGameFolder(url) {
                url = url.deletingLastPathComponent()
            }
            if Self.isGameFolder(url) {
                saveBookmark(for: url)
                // Use the bookmark-resolved URL so access outlasts the panel.
                return restoreGameFolder() ?? url
            }
            let alert = NSAlert()
            alert.messageText = "No game data found"
            alert.informativeText = "\(url.path) does not contain a baseq3 folder with .pk3 files."
            alert.runModal()
        }
        return nil
    }

    private func saveBookmark(for url: URL) {
        if let data = try? url.bookmarkData(options: [.withSecurityScope, .securityScopeAllowOnlyReadAccess],
                                            includingResourceValuesForKeys: nil,
                                            relativeTo: nil) {
            UserDefaults.standard.set(data, forKey: Self.bookmarkKey)
        }
    }
}
