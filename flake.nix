{
  description = "samidare-lib development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
      ...
    }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "samidare-lib";

        packages = [
          pkgs.python313
          pkgs.uv
          pkgs.jdk17
        ];

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
        ];

        shellHook = ''
          export JAVA_HOME="${pkgs.jdk17}"
          export UV_PYTHON="${pkgs.python313}/bin/python"
        '';
      };
    };
}
