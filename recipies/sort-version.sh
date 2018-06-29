package: sort-version
version: "%(commit_hash)s"
tag: master
source: https://github.com/frankborty/tracking_boost.git
requires:
  - boost
build_requires:
  - CMake
incremental_recipe: |
  make ${JOBS:+-j$JOBS} install
  mkdir -p $INSTALLROOT/etc/modulefiles && rsync -a --delete etc/modulefiles/ $INSTALLROOT/etc/modulefiles
---
#!/bin/bash -e

#cmake -DTRACKINGITSU_TARGET_DEVICE=OPEN_CL -DCMAKE_INSTALL_PREFIX=$INSTALLROOT $SOURCEDIR
cmake -G"Eclipse CDT4 - Unix Makefiles" -DTRACKINGITSU_TARGET_DEVICE=OPEN_CL -DCMAKE_INSTALL_PREFIX=$INSTALLROOT $SOURCEDIR \
   

make ${JOBS+-j $JOBS}
make install


# Modulefile
mkdir -p etc/modulefiles
cat > etc/modulefiles/$PKGNAME <<EoF
#%Module1.0
proc ModulesHelp { } {
  global version
  puts stderr "ALICE Modulefile for $PKGNAME $PKGVERSION-@@PKGREVISION@$PKGHASH@@"
}
set version $PKGVERSION-@@PKGREVISION@$PKGHASH@@
module-whatis "ALICE Modulefile for $PKGNAME $PKGVERSION-@@PKGREVISION@$PKGHASH@@"
# Dependencies
module load BASE/1.0 ${BOOST_VERSION:+boost/$BOOST_VERSION-$BOOST_REVISION}
# Our environment
set osname [uname sysname]
setenv TRACKINGITSU_ROOT \$::env(BASEDIR)/$PKGNAME/\$version
prepend-path PATH \$::env(TRACKINGITSU_ROOT)/bin
prepend-path LD_LIBRARY_PATH \$::env(TRACKINGITSU_ROOT)/lib
prepend-path ROOT_INCLUDE_PATH \$::env(TRACKINGITSU_ROOT)/include
$([[ ${ARCHITECTURE:0:3} == osx ]] && echo "prepend-path DYLD_LIBRARY_PATH \$::env(TRACKINGITSU_ROOT)/lib")
EoF
mkdir -p $INSTALLROOT/etc/modulefiles && rsync -a --delete etc/modulefiles/ $INSTALLROOT/etc/modulefiles
