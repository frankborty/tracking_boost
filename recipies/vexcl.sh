package: vexcl 
version: "%(commit_hash)s"
tag: master
source: https://github.com/ddemidov/vexcl.git
requires:
  - boost
build_requires:
  - CMake
---
#!/bin/bash -e
cmake -DBoost_NO_BOOST_CMAKE:BOOL=TRUE -DBoost_NO_SYSTEM_PATHS:BOOL=TRUE -DBOOST_ROOT=$BOOST_ROOT \
  -DUSE_LIBCPP:BOOL=TRUE -DCMAKE_INSTALL_PREFIX=$INSTALLROOT $SOURCEDIR 

make ${JOBS+-j $JOBS}
make install

# Modulefile
MODULEDIR="$INSTALLROOT/etc/modulefiles"
MODULEFILE="$MODULEDIR/$PKGNAME"
mkdir -p "$MODULEDIR"
cat > "$MODULEFILE" <<EoF
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
setenv VEXCL_ROOT \$::env(BASEDIR)/$PKGNAME/\$version
prepend-path PATH \$::env(VEXCL_ROOT)/bin
prepend-path LD_LIBRARY_PATH \$::env(VEXCL_ROOT)/lib
prepend-path ROOT_INCLUDE_PATH \$::env(VEXCL_ROOT)/include
$([[ ${ARCHITECTURE:0:3} == osx ]] && echo "prepend-path DYLD_LIBRARY_PATH \$::env(VEXCL_ROOT)/lib")
EoF
