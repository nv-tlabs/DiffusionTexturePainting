-- Use folder name to build extension name and tag.
local ext = get_current_extension_info()

project_ext (ext)

    -- Include pip packages installed at build time
    repo_build.prebuild_link {
        { "%{root}/_build/target-deps/pip_prebundle", ext.target_dir.."/pip_prebundle" },
    }

    -- Link only those files and folders into the extension target directory
    repo_build.prebuild_link { "docs", ext.target_dir.."/docs" }
    repo_build.prebuild_link { "data", ext.target_dir.."/data" }
    repo_build.prebuild_link { "python", ext.target_dir.."/aitoybox/texture_painter" }
